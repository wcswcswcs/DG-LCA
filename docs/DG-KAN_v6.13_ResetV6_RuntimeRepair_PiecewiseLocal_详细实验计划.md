# DG-KAN v6.13：ResetV5 Runtime Bottleneck Closure、Fused Low-Rank / Vectorized Grouped Mixing 与 Piecewise Local Primitive 实验计划

> 本计划基于 v6.12 final real-only run 制定。v6.12 已经证明：DWM2 主线继续冻结；ResetV5 family 有真实 memory 正信号，尤其 `G2-grouped-g16-poly1` 的 memory ratio mean 已到 `1.0460`，`L6-lowrank-r2-fast` 的 memory ratio mean 为 `1.0786`；但所有 measured ResetV5 candidates 都没有 memory/time 同时 near-pass。当前最大 blocker 不是 residual effect，不是 gradient correctness，也不是 fake/proxy；而是 **runtime 过慢、group/chunk/low-rank mixing 的实现路径碎片化、低层 kernel/allocation counter 不足、piecewise local primitive 尚未实现**。因此 v6.13 不继续 DWM2 patch，不打开 task / optimizer / functional correction，而是专门把 ResetV5 的 runtime bottleneck 打穿，或者给出 reset family 不可继续的路线结论。

---

## 0. 实验整体目标

v6.13 的整体目标是回答下面三个问题。

第一个问题：

$$
\boxed{
\text{ResetV5 的 memory-only 信号能否通过 runtime repair 变成 memory/time survivor？}
}
$$

第二个问题：

$$
\boxed{
\text{low-rank / grouped / chunked mixing 的 step time 失败，究竟来自数学复杂度，还是来自 Python/Torch loop、小 kernel、layout/materialization？}
}
$$

第三个问题：

$$
\boxed{
\text{如果 low-rank/grouped 路线不能同时过 memory/time，piecewise local bounded primitive 是否能成为新主线？}
}
$$

v6.12 的真实基线如下：

```text
DWM2-current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.9949 in P2
  step ratio mean   = around 1.93-2.02 depending stage
  DWM2 patch remains frozen

L6-lowrank-r2-fast:
  memory ratio mean = 1.0786
  step ratio mean   = 2.0998
  backward ratio mean = 1.3601
  memory improvement vs DWM2 = 16.51%
  residual pass = 18/18
  near pass = 0

L4-lowrank-no-intermediate:
  memory ratio mean = 1.0822
  step ratio mean   = 2.1064
  backward ratio mean = 1.3631
  residual pass = 18/18
  near pass = 0

G2-grouped-g16-poly1:
  memory ratio mean = 1.0460
  step ratio mean   = 10.1243
  backward ratio mean = 8.5911
  memory improvement vs DWM2 = 19.03%
  residual pass = 18/18
  near pass = 0
```

v6.12 的最终 route 是：

```text
R6-ResetV5TooSlow
```

这说明本轮已经不是“没有任何 memory 方向”，而是：

$$
\boxed{
\text{memory 接近 near-pass，但 runtime 完全没有过线。}
}
$$

v6.13 的最低工程成功标准是找到至少一个真实 candidate 满足：

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

终极目标仍然是：

$$
r_{\text{mem}}\leq0.80,
$$

但 v6.13 不把 $0.80$ 作为唯一 gate。v6.13 的现实目标是：判断 ResetV5 family 是否能接替 DWM2 成为新的 graph-free PureKAN lightweight primitive 主线。

---

## 1. 当前实验进展与问题判断

### 1.1 已经确定的正向进展

v6.12 与 v6.11 memory baseline 可比，DWM2 current memory ratio mean 仍为 `1.2918`。所有 measured ResetV5 candidates 继续保持：

```text
fake_data_used = 0
proxy_rows_used = 0
nonKAN_param_count = 0
manual_backward_available = 1
gradient correctness pass
residual effect pass for measured reset candidates
```

这说明当前不是实验污染、不是 non-KAN 参数混入、不是 manual gradient 公式错误。

更重要的是，ResetV5 确实出现了 memory 方向：

```text
L6-lowrank-r2-fast:
  memory ratio mean = 1.0786

G2-grouped-g16-poly1:
  memory ratio mean = 1.0460
```

这比 DWM2 current 的 `1.2918` 明显好。尤其 `G2` 已经非常接近 memory near-pass gate $1.05$。

### 1.2 当前失败的真实来源

v6.12 的失败不是 memory 完全没有方向，而是 runtime 不可接受：

```text
G2-grouped-g16-poly1:
  step ratio mean = 10.1243
  backward ratio mean = 8.5911

L6-lowrank-r2-fast:
  step ratio mean = 2.0998
  backward ratio mean = 1.3601
```

这说明分支有两类失败：

1. **Low-rank branch**：memory 还差一点，step 仍太慢，但比较平衡，值得优先修 runtime。
2. **Grouped branch**：memory 几乎 near-pass，但 step 爆炸，强烈怀疑是 Python/Torch group loop、小 kernel、grouped backward fragmentation，而不是数学本身必须这么慢。

### 1.3 v6.12 的主要未完成项

v6.12 仍有三个关键未完成点：

```text
1. P1 component audit 缺少真实低层 kernel/allocation counter；
2. piecewise local bounded reset 全部 not_implemented；
3. fused low-rank forward/backward、one-buffer、custom kernel 等关键 runtime repair 未实现。
```

因此 v6.13 的关键不是继续调 residual scale，也不是打开 task，而是先补齐 runtime diagnosis 和实现真正 fused/vectorized path。

### 1.4 是否在正确道路上

项目整体仍在正确道路上。原因是：

```text
DWM2 已经按 route 冻结，没有继续小修；
ResetV5 出现真实 memory 改善；
residual effect 已经过 gate；
task/optimizer/functional 继续被正确 gate；
没有把 memory-only signal 写成 task success。
```

但 v6.13 必须避免一个风险：不要把 `G2` 的 memory ratio `1.0460` 误读为已经接近成功。它的 step ratio 是 `10.1243`，这不是 near miss，而是 severe runtime failure。

因此当前路线应该是：

$$
\boxed{
\text{围绕 ResetV5 做 runtime-first 修复，而不是 memory-only 搜索。}
}
$$

### 1.5 离目标还差多远

对于最平衡 low-rank candidate `L6-lowrank-r2-fast`：

$$
r_{\text{mem}}=1.0786.
$$

到 near-pass $1.05$ 需要降低：

$$
1-\frac{1.05}{1.0786}\approx2.65\%.
$$

到 formal memory pass $1.00$ 需要降低：

$$
1-\frac{1.00}{1.0786}\approx7.29\%.
$$

到终极 $0.80$ 需要降低：

$$
1-\frac{0.80}{1.0786}\approx25.82\%.
$$

它的 step ratio 为：

$$
r_{\text{step}}=2.0998.
$$

到 near-pass $1.50$ 需要提速：

$$
1-\frac{1.50}{2.0998}\approx28.56\%.
$$

到 task-open $1.35$ 需要提速：

$$
1-\frac{1.35}{2.0998}\approx35.71\%.
$$

对于 memory 最好的 `G2-grouped-g16-poly1`：

$$
r_{\text{mem}}=1.0460.
$$

它已经达到 memory near-pass，但 step ratio 是：

$$
r_{\text{step}}=10.1243.
$$

到 near-pass $1.50$ 需要提速：

$$
1-\frac{1.50}{10.1243}\approx85.18\%.
$$

这说明 grouped branch 不是简单微调，而是必须彻底改实现方式。

---

## 2. v6.13 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许继续 DWM2 局部 patch。DWM2 只作为 frozen baseline，不作为 v6.13 主线。

第三，不允许只看 memory ratio。任何 candidate 若 step ratio $>1.50$，不能进入 P7 one-step probe，更不能进入 task。

第四，不允许只调 residual scale。v6.11/v6.12 已证明 residual effect 可以过 gate。v6.13 的修复对象是 runtime / mixing / workspace model。

第五，不允许把 `G2-grouped-g16-poly1` 写成 near-pass survivor。它是 memory-near but runtime-fail candidate。

第六，不允许把 P1 full-step phase fields 当作低层 kernel counter。v6.13 必须明确区分：

```text
phase-level diagnostic
kernel-level profiler
allocation-level profiler
```

第七，如果 low-level kernel/allocation counter 不可用，必须写 `metric_unavailable`，不能补假数。

第八，没有 S0/S1/S2 survivor 时，不允许打开：

```text
task re-entry
optimizer exploration
functional correction
LightSmooth
multi-seed confirm
```

---

## 3. 核心假设

### H1：Low-rank branch 已接近 memory near-pass，主要 blocker 是 runtime fragmentation

`L6-lowrank-r2-fast` memory ratio mean 为 $1.0786$，离 $1.05$ 只差约 $2.65\%$，但 step ratio mean 为 $2.0998$。H1 假设：low-rank 的 runtime 失败来自 unfused low-rank projection、intermediate materialization、小 kernel、allocation overhead，而不是 low-rank 数学本身过重。

H1 成立需要 P1/P2 观察到：

```text
lowrank_projection_time_fraction >= 0.25
or lowrank_kernel_count is high
or lowrank_intermediate_MB is significant
or allocation_count_inside_lowrank is high
```

H1 的修复目标是：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75,
$$

同时：

$$
\frac{M_{\text{new}}}{M_{\text{L6}}}\leq1.05.
$$

如果满足，则 low-rank branch 继续作为主线。

---

### H2：Grouped branch 的 memory 信号是真实的，但 runtime 爆炸来自 Python/Torch group loop

`G2-grouped-g16-poly1` memory ratio mean 为 $1.0460$，但 step ratio mean 为 $10.1243$。H2 假设：grouped branch 的数学计算不是必然 $10\times$ 慢，主要问题是 group loop 被拆成过多小 op / 小 kernel / backward group loop。

H2 成立需要观察到：

```text
group_loop_count >= group_count
kernel_count scales with group_count
step_time increases roughly linearly with group_count
batched grouped implementation reduces step by >=50%
```

H2 的修复目标是：

$$
\frac{T_{\text{step,vectorized-grouped}}}{T_{\text{step,G2}}}\leq0.30.
$$

同时保持：

$$
r_{\text{mem}}\leq1.05.
$$

如果 vectorized grouped 仍然 $>3\times$ MLP，则 grouped branch 只作为 memory diagnostic，不作为主线。

---

### H3：Chunked mixing 降 memory 但增加 kernel/loop overhead，需要找到 Pareto 而不是最低 memory

v6.12 中 chunk size 越小，step 越差。H3 假设存在 chunk size $c^\*$，使 memory 接近 $1.05$，同时 step 不超过 $1.50$。

候选：

```text
chunk_size = 16, 32, 64, 128
rank = 2, 4
```

H3 成立要求存在 candidate 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果没有这样的 point，但 Pareto 曲线清楚显示 memory-time tradeoff，则该分支可以继续一轮 runtime repair；否则停止 chunked branch。

---

### H4：Piecewise local bounded reset 尚未被验证，不能被判失败

v6.12 的 piecewise local bounded reset 全部是 `not_implemented`，因此不能说这条路线失败。H4 假设：piecewise local residual 可以在不引入 dense basis tensor 的情况下保持 residual effect，并且比 low-rank/grouped 更适合 bounded workspace。

H4 成立要求：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02,
$$

并且：

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

如果 forwardOnly diagnostic 过 memory/time，但 streamingGrad 后失败，则说明 knot gradient accumulation 是 blocker，需要单独修。

---

### H5：当前 P1 低层 counter 不足会阻碍 runtime 修复

v6.12 failure table 中 `F5_low_level_counter_unavailable=720`。H5 假设：如果不拿到 kernel/allocation counters，runtime repair 仍会盲打。

H5 成立标准是 v6.13 P1 至少对 best low-rank / grouped candidates 取得：

```text
component_kernel_count
component_allocation_count
component_gemm_count
component_elementwise_count
component_custom_kernel_count
```

如果这些仍不可用，则 route 必须标记：

```text
R3-RuntimeAttributionIncomplete
```

而不能宣称某个 runtime repair 已完成。

---

### H6：如果 v6.13 仍无 S2 survivor，应进入全新 primitive family，而不是继续 ResetV5 patch

H6 是 Stop/Go 假设。如果本轮完成：

```text
low-rank runtime repair
grouped vectorization
chunked Pareto
piecewise local implementation
kernel/allocation audit
```

仍没有 S2 survivor，则应输出：

```text
R7-NoViableResetV5
```

下一轮必须进入新 primitive family 设计。

---

## 4. 实验阶段总览

v6.13 分为十二个阶段：

```text
P0: v6.12 reproduction and reset-v5 contract
P1: reset-v5 low-level runtime attribution
P2: fused low-rank runtime repair v2
P3: vectorized grouped mixing repair
P4: chunked low-rank Pareto refinement
P5: piecewise local bounded reset implementation
P6: reset-v6 package selection
P7: one-step loss and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P6 是核心。P7 只对 S0/S1/S2 candidate 运行。P8-P10 默认关闭。

---

## 5. P0：v6.12 reproduction and reset-v5 contract

### 5.1 目的

确认 v6.13 与 v6.12 baseline 可比，并确认 DWM2 仍冻结、reset lineage 真实。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-frozen-baseline
L6-lowrank-r2-fast
L4-lowrank-no-intermediate
G2-grouped-g16-poly1
G1-grouped-g8-poly1
ResetV6-fused-lowrank-r2, if implemented
ResetV6-vectorized-grouped-g16, if implemented
ResetV6-piecewise2-streamingGrad, if implemented
```

### 5.3 必须记录字段

```text
variant
family
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
residual_param_count
mixing_param_count
rollback_max_error
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
v612_memory_ratio_mean
v613_memory_ratio_mean
v612_step_ratio_mean
v613_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
is_dwm2_patch
dwm2_patch_allowed
```

### 5.4 通过标准

Reproduction pass:

$$
|r_{\text{mem,v613}}-r_{\text{mem,v612}}|\leq0.05.
$$

$$
|r_{\text{step,v613}}-r_{\text{step,v612}}|\leq0.15.
$$

Contract pass:

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
manual_backward_available = 1
rollback_max_error < 1e-8
grad_relerr < 1e-4
grad_cos > 0.999
```

DWM2 freeze pass:

```text
is_dwm2_patch = 0 for all new measured candidates
```

### 5.5 可视化

```text
p0_reproduction_delta_bar.svg
p0_contract_heatmap.svg
p0_reset_lineage_table.md
```

---

## 6. P1：reset-v5 low-level runtime attribution

### 6.1 目的

P1 要补齐 v6.12 的最大诊断缺口：真实 kernel/allocation counter。v6.12 的 component audit 是 full-step phase fields，不是底层 kernel counter。v6.13 必须明确 runtime 为什么慢。

### 6.2 测量对象

```text
L6-lowrank-r2-fast
L4-lowrank-no-intermediate
G2-grouped-g16-poly1
G1-grouped-g8-poly1
C5-lowrank-r4-chunk64
DWM2-current-frozen-baseline
MLP-manual-linear-reference
```

### 6.3 Component phases

```text
phase_residual_transform
phase_lowrank_factor_V
phase_lowrank_factor_U
phase_group_loop_forward
phase_group_loop_backward
phase_chunked_mixing
phase_loss_delta
phase_backward_mixing_delta
phase_backward_residual_dx
phase_backward_residual_params
phase_update_prep
phase_optimizer_update
```

### 6.4 必须记录字段

```text
variant
family
dataset
batch_size
depth
component
component_time_ms
component_peak_MB
component_alloc_count
component_kernel_count
component_gemm_count
component_elementwise_count
component_custom_kernel_count
component_temp_MB
component_fraction_of_step_time
component_fraction_of_peak_gap
group_loop_count
chunk_loop_count
python_loop_count
torch_op_count
cuda_sync_count
global_load_bytes
global_store_bytes
l2_read_transactions
l2_write_transactions
stall_long_scoreboard
metric_availability
grad_relerr_max
grad_cos_min
```

如果 Nsight / kernel counter 不可用，必须写：

```text
metric_unavailable
```

不能填假值。

### 6.5 判断标准

P1 runtime attribution pass 要求：

```text
component_kernel_count available
component_alloc_count available
or clear torch_op_count/python_loop_count available
```

并且至少识别一个 primary runtime blocker：

```text
primary_runtime_blocker in {
  lowrank_factor_materialization,
  group_loop_fragmentation,
  chunk_loop_fragmentation,
  mixing_workspace,
  backward_mixing_delta,
  residual_dx,
  optimizer_update,
  unknown
}
```

若：

```text
primary_runtime_blocker = unknown
```

则 P2-P5 仍可做 implementation diagnostic，但 route 必须标记 runtime attribution incomplete。

### 6.6 可视化

```text
p1_component_runtime_waterfall.svg
p1_component_memory_waterfall.svg
p1_component_kernel_count_bar.svg
p1_group_loop_scaling_plot.svg
p1_lowrank_factor_runtime_bar.svg
p1_runtime_blocker_heatmap.svg
```

---

## 7. P2：fused low-rank runtime repair v2

### 7.1 目的

P2 以 `L6-lowrank-r2-fast` 和 `L4-lowrank-no-intermediate` 为主线，优先修 step time，同时保持 memory improvement。

### 7.2 Candidate packages

```text
LR0-L6-lowrank-r2-fast-current
LR1-r2-fused-forward
LR2-r2-fused-backward
LR3-r2-fused-forward-backward
LR4-r2-no-materialized-lowrank-intermediate
LR5-r2-onebuffer-lowrank
LR6-r2-vectorized-lowrank
LR7-r2-triton-lowrank-forward
LR8-r2-triton-lowrank-backward
LR9-r2-full-fused-lowrank
```

如果 r2 capacity 被认为太低，可加：

```text
LR10-r4-full-fused-lowrank
```

但 r4 不是第一优先级。

### 7.3 必须记录字段

```text
package
rank
implementation_status
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_improvement_vs_L6
step_improvement_vs_L6
memory_improvement_vs_DWM2
step_improvement_vs_DWM2
intermediate_lowrank_MB
lowrank_kernel_count
lowrank_allocation_count
lowrank_temp_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 7.4 判断标准

Runtime repair pass relative to L6:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75.
$$

Memory no-regression:

$$
\frac{M_{\text{new}}}{M_{\text{L6}}}\leq1.05.
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

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

### 7.5 可视化

```text
p2_lowrank_package_pareto.svg
p2_step_improvement_waterfall.svg
p2_lowrank_intermediate_memory_bar.svg
p2_rank_vs_memory_time.svg
p2_lowrank_kernel_count_bar.svg
```

---

## 8. P3：vectorized grouped mixing repair

### 8.1 目的

P3 解决 grouped branch 的巨大 step time。`G2-grouped-g16-poly1` memory ratio 已经 near-pass，但 step ratio 极差。P3 要验证 vectorized grouped implementation 是否能把 group loop overhead 消掉。

### 8.2 Candidate packages

```text
GR0-G2-grouped-g16-current
GR1-g16-vectorized-forward
GR2-g16-vectorized-backward
GR3-g16-vectorized-forward-backward
GR4-g16-batched-group-gemm
GR5-g16-single-kernel-grouped-mix
GR6-g8-vectorized-forward-backward
GR7-g4-vectorized-forward-backward
GR8-g16-triton-grouped-mix
GR9-g16-triton-grouped-backward
```

### 8.3 必须记录字段

```text
package
group_count
implementation_status
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_improvement_vs_G2
step_improvement_vs_G2
memory_improvement_vs_DWM2
step_improvement_vs_DWM2
group_loop_count
group_kernel_count
group_allocation_count
batched_group_gemm_count
group_workspace_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 8.4 判断标准

Grouped runtime repair useful:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,G2}}}\leq0.30.
$$

Near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

If step still:

$$
r_{\text{step}}>3.0,
$$

then grouped branch is classified as:

```text
memory-only diagnostic, not mainline
```

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

### 8.5 可视化

```text
p3_grouped_runtime_pareto.svg
p3_group_count_vs_step.svg
p3_group_loop_count_vs_step.svg
p3_vectorized_vs_loop_bar.svg
p3_group_workspace_memory_bar.svg
```

---

## 9. P4：chunked low-rank Pareto refinement

### 9.1 目的

P4 重新测 chunked low-rank，但重点不是继续压 memory，而是找 Pareto。v6.12 已经显示 chunk 越小 step 越差，所以 v6.13 只测可能合理的 chunk size，不再浪费在明显过小 chunk。

### 9.2 Candidate packages

```text
CH0-L6-r2-current
CH1-r2-chunk32
CH2-r2-chunk64
CH3-r2-chunk128
CH4-r4-chunk64
CH5-r4-chunk128
CH6-r2-adaptive-chunk
CH7-r4-adaptive-chunk
```

### 9.3 必须记录字段

```text
package
rank
chunk_size
adaptive_chunk_enabled
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
mixing_workspace_MB
chunk_boundary_overhead_ms
chunk_kernel_count
chunk_allocation_count
memory_improvement_vs_L6
step_improvement_vs_L6
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 9.4 判断标准

Chunked Pareto useful:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

If no candidate satisfies both but one achieves:

$$
r_{\text{mem}}\leq1.05
$$

and

$$
r_{\text{step}}\leq2.0,
$$

then it is diagnostic only.

### 9.5 可视化

```text
p4_chunk_size_pareto.svg
p4_chunk_size_memory_curve.svg
p4_chunk_size_step_curve.svg
p4_rank_chunk_heatmap.svg
```

---

## 10. P5：piecewise local bounded reset implementation

### 10.1 目的

P5 是 v6.13 的新 primitive branch。v6.12 piecewise 全部 `not_implemented`，因此本轮必须至少实现一个 minimal measured candidate，不能继续空缺。

### 10.2 Candidate packages

```text
PW0-piecewise2-forwardOnly-diagnostic:
  forward + input backward only, no knot grad

PW1-piecewise2-streamingGrad:
  adds streaming knot grad

PW2-piecewise4-streamingGrad:
  4 local knots

PW3-piecewise2-groupedMix:
  piecewise residual + grouped mixing

PW4-piecewise2-lowrankMix-r2:
  piecewise residual + lowrank r2 mixing

PW5-piecewise2-no-knot-materialization:
  avoid full [B,d,K] basis tensor
```

### 10.3 必须记录字段

```text
package
num_bins
active_bins_per_sample
streaming_grad_enabled
mixing_type
knot_materialized
dense_basis_tensor_used
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
knot_workspace_MB
knot_grad_workspace_MB
mixing_workspace_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
bin_occupancy_entropy
dead_bin_fraction
out_of_grid_fraction
```

### 10.4 判断标准

Structural pass:

```text
dense_basis_tensor_used = 0
active_bins_per_sample <= 4
manual_backward_available = 1
```

Efficiency near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

Residual effect:

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02.
$$

Stability:

$$
\operatorname{out\_of\_grid\_fraction}\leq0.05.
$$

$$
\operatorname{dead\_bin\_fraction}\leq0.30.
$$

If forwardOnly passes but streamingGrad fails, route should identify knot gradient accumulation as blocker.

### 10.5 可视化

```text
p5_piecewise_pareto.svg
p5_bin_occupancy_heatmap.svg
p5_dead_bin_fraction_bar.svg
p5_out_of_grid_curve.svg
p5_knot_workspace_bar.svg
```

---

## 11. P6：reset-v6 package selection

### 11.1 目的

P6 汇总 P2-P5，判断是否存在 S0/S1/S2 survivor。

### 11.2 Survivor 类型

```text
S0:
  r_mem < 1.0 and r_step <= 1.20

S1:
  r_mem < 1.0 and r_step <= 1.35

S2:
  r_mem <= 1.05 and r_step <= 1.50 and residual effect pass

S3:
  memory near-pass but step fails

S4:
  step improves but memory fails

S5:
  residual effect fails

S6:
  gradient correctness fails

S7:
  no viable reset-v6
```

### 11.3 必须记录

```text
best_candidate
best_family
best_memory_ratio
best_step_ratio
best_backward_ratio
best_forward_ratio
memory_improvement_vs_current
step_improvement_vs_current
residual_effect_pass
grad_pass
survivor_type
open_one_step_probe
open_task_reentry
primary_blocker
```

### 11.4 判断标准

Open P7 one-step probe if:

```text
survivor_type in {S0, S1, S2}
```

Open official task re-entry only if:

```text
survivor_type in {S0, S1}
```

No task if:

```text
survivor_type in {S3, S4, S5, S6, S7}
```

### 11.5 可视化

```text
p6_reset_v6_scorecard.svg
p6_family_comparison_pareto.svg
p6_survivor_type_dashboard.svg
```

---

## 12. P7：one-step loss and residual probe

P7 只对 P6 S0/S1/S2 运行。

### 12.1 方法

```text
clone weights
compute manual gradient
apply one update
measure train loss before/after
measure holdout loss before/after
measure val loss before/after
measure residual ablation before/after
rollback weights
check rollback exactness
```

### 12.2 必须记录

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
p7_loss_before_after.svg
p7_bad_step_heatmap.svg
p7_update_norm_vs_loss_delta.svg
p7_residual_ablation_probe.svg
```

---

## 13. P8：limited task re-entry gate

P8 remains closed unless P7 passes and memory/time gates hold.

### 13.1 打开条件

Official task:

```text
survivor_type in {S0, S1}
P7 pass
r_mem < 1.0
r_step <= 1.35
grad pass
residual effect pass
no fake/proxy
```

Diagnostic-only task:

```text
survivor_type == S2
P7 pass
r_mem <= 1.05
r_step <= 1.50
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
  Best-reset-v6 + ManualAdanLite
  Best-reset-v6 + ManualAdamW
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
feature_rank
margin_mean
margin_p10
seedwise_failure_reason
```

### 13.4 判断标准

Task pass:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets.

At least two datasets:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

Memory must hold:

$$
r_{\text{mem}}<1.0.
$$

Wall-clock must hold on at least two datasets:

$$
\operatorname{ValLossAUC}_{time,KAN}
\leq
\operatorname{ValLossAUC}_{time,MLP}.
$$

### 13.5 可视化

```text
p8_val_loss_vs_step.svg
p8_val_loss_vs_time.svg
p8_accuracy_vs_time.svg
p8_time_to_target_bar.svg
p8_task_efficiency_pareto.svg
p8_seedwise_delta_plot.svg
```

---

## 14. P9：optimizer exploration remains gated

P9 只在 P8 official task pass 后打开。它不是 v6.13 主线。

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

P10 只在 P8/P9 pass 后打开。默认 `not_run`。

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

## 16. P11：route decision

### Route cases

```text
R1-ResetV6Solved:
  reset-v6 gets S0/S1 and P7 pass.
  Open task.

R2-ResetV6NearPass:
  reset-v6 gets S2.
  Continue reset engineering, diagnostic task only.

R3-LowRankRuntimeRepairNeeded:
  low-rank memory is close but step still fails.
  Continue low-rank runtime fusion.

R4-GroupedRuntimeFail:
  grouped memory is good but runtime remains too slow.
  Stop grouped branch unless vectorized kernel exists.

R5-PiecewiseLocalCandidate:
  piecewise local reset gets S0/S1/S2.
  Switch mainline to piecewise local.

R6-RuntimeAttributionIncomplete:
  kernel/allocation counters still unavailable.
  Improve profiler before making route claim.

R7-NoViableResetV6:
  no reset-v6 candidate gets S2.
  Need new primitive family.

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
best_forward_ratio
memory_improvement_vs_current
step_improvement_vs_current
survivor_type
residual_effect_pass
grad_pass
runtime_attribution_pass
open_one_step_probe
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
F4_residual_effect_fail
F5_lowrank_runtime_fail
F6_grouped_runtime_fail
F7_chunked_time_fail
F8_piecewise_stability_fail
F9_runtime_counter_unavailable
F10_task_gated
F11_optimizer_gated
F12_functional_gated
F13_fake_or_proxy_violation
F14_artifact_missing
```

### artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_reset_runtime_attribution.csv
p2_fused_lowrank_repair_v2.csv
p3_vectorized_grouped_repair.csv
p4_chunked_lowrank_pareto.csv
p5_piecewise_local_reset.csv
p6_reset_v6_selection.csv
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

### figures

```text
figures/p1_component_runtime_waterfall.svg
figures/p1_component_kernel_count_bar.svg
figures/p2_lowrank_package_pareto.svg
figures/p3_grouped_runtime_pareto.svg
figures/p4_chunk_size_pareto.svg
figures/p5_piecewise_pareto.svg
figures/p6_family_comparison_pareto.svg
figures/p7_loss_before_after.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：low-rank branch 达到 S0/S1/S2

如果 fused low-rank / vectorized low-rank 达到 S2 或更好，reset-v6 继续作为主线。若达到 S0/S1，打开 P7/P8。

### Case B：low-rank memory 好但 step 仍失败

如果 memory $\leq1.05$，但 step $>1.50$，说明 low-rank memory model 有用但 runtime 不行。下一步继续 low-rank fusion，不开 task。

### Case C：grouped branch step 仍极慢

如果 grouped vectorized 后 step 仍 $>3.0$，则 grouped branch 停止作为主线，仅保留 memory diagnostic。

### Case D：piecewise local 成功

如果 piecewise local reset 达到 S0/S1/S2 且 stability pass，则切换到 piecewise local。

### Case E：所有 reset-v6 失败

如果没有 S2 candidate，则停止当前 ResetV5/V6 分支，进入 new primitive family design。

---

## 19. 最终建议

v6.13 的一句话策略是：

$$
\boxed{
\text{以 ResetV5 的 memory signal 为基础，优先修 runtime；同时首次真实实现 piecewise local bounded primitive。}
}
$$

DWM2 继续冻结。Low-rank 是最平衡起点，grouped 是 memory-only diagnostic，piecewise local 是必须补齐的新 primitive branch。

如果 v6.13 仍然没有 S0/S1/S2 candidate，就应该正式进入新的 primitive family 设计，不再围绕当前 low-rank/grouped/poly1 reset 做局部 patch。
