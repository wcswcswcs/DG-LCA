# DG-KAN v6.14：ResetV6 Runtime Counter Closure、Memory-Preserving Grouped Fusion 与 Low-Rank/Piecewise 分支决策实验计划

> 本计划基于 v6.13 final real-only run 制定。v6.13 没有得到 S0/S1/S2 survivor，但它给出了非常清楚的新路线信号：DWM2 继续冻结；ResetV6 family 中 grouped branch 给出最强 memory signal，`G2-grouped-g16-poly1` 的 memory ratio mean 已到 `1.0460`，但 step ratio mean 高达 `9.7583`；vectorized grouped branch 将 step ratio 降到约 `1.98`，但 memory ratio 回退到 `1.1749`；low-rank r2 branch 更平衡，memory ratio mean `1.0786`、step ratio mean `2.0141`；piecewise local branch 已真实实现并通过 structural/stability/gradient/residual gates，但 memory/time 明显失败。因此 v6.14 的目标不是继续 DWM2 patch，也不是打开 task / optimizer / functional correction，而是完成 ResetV6 的 runtime counter 闭环，并验证是否能把 grouped 的 memory 优势和 vectorized runtime 优势合并到同一个 candidate 中。

---

## 0. 实验整体目标

v6.14 的整体目标是回答三个问题。

第一个问题：

$$
\boxed{
\text{ResetV6 grouped branch 的 step 爆炸到底来自 Python group loop、小 kernel、materialization，还是 grouped 数学本身？}
}
$$

第二个问题：

$$
\boxed{
\text{能否保留 } G2 \text{ 的 memory ratio } \approx 1.046 \text{，同时获得 } GR3/GR4 \text{ 的 vectorized runtime？}
}
$$

第三个问题：

$$
\boxed{
\text{如果 grouped branch 不能同时过 memory/time，low-rank r2 或 piecewise local 是否能成为新的主线？}
}
$$

v6.13 的真实基线如下：

```text
DWM2-current-frozen-baseline:
  memory ratio mean = 1.2918
  step ratio mean   = 1.8658 to 1.9949 depending stage
  DWM2 remains frozen

L6-lowrank-r2-fast:
  memory ratio mean  = 1.0786
  step ratio mean    = 1.9968 in P1 / 2.0141 in P2
  backward ratio mean = 1.2396
  residual effect pass = 18/18
  near pass = 0

G2-grouped-g16-poly1:
  memory ratio mean  = 1.0460
  step ratio mean    = 9.5095 in P1 / 9.7583 in P6
  backward ratio mean = 7.7978 to 8.5911
  residual effect pass = 18/18
  near pass = 0

GR3-g16-vectorized-forward-backward:
  memory ratio mean  = 1.1749
  step ratio mean    = 1.9291 in P1 / 1.9817 in P3
  backward ratio mean = 1.4968
  residual effect pass = 18/18
  near pass = 0

PW0-piecewise2-forwardOnly-diagnostic:
  memory ratio mean = 1.3277
  step ratio mean   = 3.7215
  structural/stability/residual pass = 1

PW1-piecewise2-streamingGrad:
  memory ratio mean = 1.4862
  step ratio mean   = 4.0981
  structural/stability/residual pass = 1
```

v6.13 的最终 route 是：

```text
R4-GroupedRuntimeFail
primary_blocker = runtime_too_slow
next_required_implementation = fused_grouped_kernel_or_stop_grouped_branch
```

v6.14 的最低工程成功标准是找到至少一个真实 candidate 满足：

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
r_{\text{mem}}\leq0.80.
$$

v6.14 不以 $0.80$ 作为唯一 gate。v6.14 的现实目标是：判断 ResetV6 的 grouped / low-rank / piecewise 三个分支中，是否存在一个可以进入 one-step probe 或 limited task re-entry 的真实 lightweight PureKAN candidate。

---

## 1. 当前实验进展与问题判断

### 1.1 已经确定的正向进展

v6.13 继续保持 real-only 纪律。P0-P10 的 `fake_data_used` 与 `proxy_row_used/proxy_rows_used` 总和均为 `0`，DWM2 继续作为 frozen baseline，没有新增 DWM2 patch。所有 measured ResetV6 candidates 保持：

```text
nonKAN_param_count = 0
manual_backward_available = 1
gradient correctness pass
residual effect pass for measured reset candidates
```

这说明当前不需要回头怀疑：

```text
graph-free manual backward
strict PureKAN accounting
residual effect
fake/proxy contamination
```

当前确实是在 lightweight primitive 的 memory/time 层面失败。

### 1.2 当前最重要的新信号

v6.13 给出了两个互补信号。

第一个是 memory signal：

```text
G2-grouped-g16-poly1:
  memory ratio mean = 1.0460
```

它已经达到 memory near-pass 的数值区域。相比 DWM2-current 的 `1.2918`，memory improvement 约为 `19.03%`。

第二个是 runtime signal：

```text
GR3-g16-vectorized-forward-backward:
  step ratio mean = 1.9817
```

它相比 `G2` 的 step ratio `9.7583` 提速约 `80%`，证明 grouped branch 的 $10\times$ 慢不是数学必然，而主要来自实现方式。

但这两个信号没有在同一个 candidate 上同时成立：

```text
G2:
  memory 好，runtime 崩

GR3/GR4:
  runtime 修复，memory 回退
```

因此 v6.14 的核心问题是：

$$
\boxed{
\text{如何把 G2 的 memory path 和 GR3/GR4 的 runtime path 合并？}
}
$$

### 1.3 目前的主要失败点

v6.13 failure table 中最大失败仍然是：

```text
F9_runtime_counter_unavailable = 1584
```

这说明虽然 P1 component phase fields 和 code-loop diagnostics 可用，但真实低层 kernel/allocation counter 仍没有闭合。当前还不能精确判断：

```text
G2 到底慢在 Python loop 还是每组小 GEMM？
GR3/GR4 memory 回退来自 batched materialization 还是 workspace layout？
low-rank step ratio 约 2.0 来自 GEMM、elementwise、layout 还是 update？
piecewise local 的 memory/time 失败来自 knot grad 还是 mixing？
```

所以 v6.14 的第一任务不是再扫更多超参，而是补齐 runtime attribution。

### 1.4 分支判断

当前三个分支的状态是：

```text
Grouped branch:
  memory 最接近目标，但 runtime 需要 fused/vectorized 且不能牺牲 memory。

Low-rank branch:
  更平衡，memory 只差约 2.65% 到 near-pass，但 step 还差约 25% 到 1.50。

Piecewise branch:
  结构与稳定性已过，首次真实实现，但 memory/time 明显失败，需要拆清楚是 residual/knot 还是 mixing。
```

v6.14 不能只押单一方向。它应该主攻 grouped 的 memory-preserving fused runtime，同时保留 low-rank 作为平衡候选，piecewise 作为结构候选。

### 1.5 离目标还差多远

#### Low-rank r2 branch

对于 `L6-lowrank-r2-fast`：

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

它的 step ratio 约为：

$$
r_{\text{step}}\approx2.01.
$$

到 near-pass $1.50$ 需要提速：

$$
1-\frac{1.50}{2.01}\approx25.37\%.
$$

到 task-open $1.35$ 需要提速：

$$
1-\frac{1.35}{2.01}\approx32.84\%.
$$

#### Grouped branch

对于 `G2-grouped-g16-poly1`：

$$
r_{\text{mem}}=1.0460.
$$

memory 已经进入 near-pass 区域。但：

$$
r_{\text{step}}=9.7583.
$$

到 near-pass $1.50$ 需要提速：

$$
1-\frac{1.50}{9.7583}\approx84.63\%.
$$

这只能靠 vectorized / fused / batched kernel，不能靠普通调参。

#### Vectorized grouped branch

对于 `GR3-g16-vectorized-forward-backward`：

$$
r_{\text{step}}\approx1.9817,
$$

到 $1.50$ 需要提速：

$$
1-\frac{1.50}{1.9817}\approx24.31\%.
$$

但它的 memory ratio 是：

$$
r_{\text{mem}}=1.1749,
$$

到 $1.05$ 需要降低：

$$
1-\frac{1.05}{1.1749}\approx10.63\%.
$$

因此 vectorized grouped 分支的目标是减少 materialization 与 workspace，而不是再做粗粒度 vectorization。

#### Piecewise branch

`PW0` 的 memory ratio 为 `1.3277`、step ratio 为 `3.7215`。这离目标较远，但它是首次实现，不能直接作为最终失败结论。它需要先拆分：

```text
piecewise residual only
piecewise + mixing
streaming knot grad
grouped/lowrank mixing
```

---

## 2. v6.14 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许继续 DWM2 局部 patch。DWM2 只作为 frozen baseline。

第三，不允许把 memory-only candidate 作为 survivor。`G2` 不能进入 one-step probe 或 task，除非 step ratio 降到 $1.50$ 以下。

第四，不允许把 vectorized grouped 的 runtime improvement 写成成功，除非 memory 不回退到 $1.05$ 以上。

第五，不允许继续只做 P1 phase fields。v6.14 必须尽量取得 kernel/allocation counter 或至少取得可靠 torch op / python loop / kernel count 代理。

第六，P7/P8/P9/P10 没有 S0/S1/S2 survivor 时必须继续关闭。

第七，piecewise local 不能因为 structural/stability 通过就进入 task，必须先通过 memory/time gate。

第八，不允许只调 chunk size、group count、rank，而不分析 runtime component。所有 sweep 必须和 runtime attribution 绑定。

---

## 3. 核心假设

### H1：Grouped branch 的 $10\times$ step time 主要来自 Python group loop 和小 kernel fragmentation

`G2-grouped-g16-poly1` memory 很好但 step 极慢。H1 假设：主要慢因是 group loop 被拆成太多小 op，而不是 grouped 数学本身。

H1 成立标准是至少满足两项：

```text
python_loop_count scales with group_count
kernel_count scales with group_count
group_loop_forward + group_loop_backward accounts for >= 50% step time
vectorized grouped reduces step time by >= 70%
```

v6.13 已经看到 vectorized grouped 约 $80\%$ step improvement，因此 H1 已部分成立。v6.14 要进一步判断 memory 回退的原因。

---

### H2：Vectorized grouped 的 memory 回退来自 grouped materialization 或 batched workspace，而不是 grouped 算法本身

`GR3/GR4` 把 step 从约 `9.76` 降到约 `1.98`，但 memory 从 `1.0460` 回退到 `1.1749`。H2 假设：memory 回退来自 vectorized implementation materialize 了额外 tensor，例如：

```text
grouped_activation_tensor
batched_group_workspace
einsum_intermediate
layout_conversion
contiguous_copy
batched_group_grad_temp
```

H2 成立标准是 attribution 显示这些 grouped materialization source 解释至少：

$$
\frac{M_{\text{grouped materialization}}}{G_{\text{peak}}}\geq0.30.
$$

H2 修复标准是 memory-preserving vectorized grouped package 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

---

### H3：Low-rank r2 是最平衡 fallback，但需要 fused runtime 与轻量 memory reduction

`L6` 的 memory 很接近 near-pass，但 step 太慢。H3 假设：通过 fused r2 low-rank forward/backward、减少 intermediate、或使用 custom kernel，可以同时获得：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

H3 的核心修复目标是相对 `L6`：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75,
$$

且：

$$
\frac{M_{\text{new}}}{M_{\text{L6}}}\leq1.05.
$$

如果不能满足，low-rank branch 保留为 memory diagnostic，但不能打开 task。

---

### H4：Piecewise local 当前失败可能来自 mixing，而不是 local residual 本身

Piecewise local structural/stability/residual effect 均通过，但 memory/time 很差。H4 假设：piecewise residual 本身未必是主因，失败可能来自和 grouped/lowrank mixing 组合后的 workspace。

H4 成立需要 component audit 显示：

```text
piecewise_residual_transform_fraction < mixing_fraction
or knot_grad_workspace < mixing_workspace
```

如果 piecewise residual-only diagnostic 很重，则 piecewise branch 暂停。如果 residual-only 轻但 mixing 重，则继续修 piecewise + bounded mixing。

---

### H5：真实 runtime counter 是决定下一步路线的必要条件

v6.13 的 `F9_runtime_counter_unavailable=1584` 是最大诊断缺口。H5 假设：如果没有 component kernel count / allocation count / op count，无法判断当前分支是否可修。

P1 runtime attribution pass 要求至少取得以下之一：

```text
component_kernel_count
component_allocation_count
torch_op_count
python_loop_count
cuda_sync_count
```

如果全部不可用，则 route 必须是：

```text
R6-RuntimeAttributionIncomplete
```

不能声明 grouped 或 low-rank 已经被证伪。

---

### H6：如果 v6.14 仍然没有 S2 survivor，应停止 ResetV6 当前分支

H6 是 Stop/Go 假设。如果完成：

```text
memory-preserving grouped fusion
low-rank runtime repair
piecewise residual/mixing split
runtime attribution
```

仍没有 S2 survivor，则当前 ResetV6 分支应停止，进入 new primitive family design。

---

## 4. 实验阶段总览

v6.14 分为十二个阶段：

```text
P0: v6.13 reproduction and contract check
P1: reset-v6 runtime attribution with kernel/op counters
P2: memory-preserving vectorized grouped fusion
P3: grouped custom/batched kernel diagnostic
P4: low-rank r2 runtime closure
P5: piecewise residual-vs-mixing decomposition
P6: reset-v7 package selection
P7: one-step loss and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P6 是核心。P7 只对 S0/S1/S2 candidate 运行。P8-P10 默认关闭。

---

## 5. P0：v6.13 reproduction and contract check

### 5.1 目的

确认 v6.14 与 v6.13 baseline 可比，并确保 DWM2 继续冻结、ResetV7 新候选是 strict PureKAN / graph-free。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-frozen-baseline
L6-lowrank-r2-fast
G2-grouped-g16-poly1
GR3-g16-vectorized-forward-backward
GR4-g16-batched-group-gemm
PW0-piecewise2-forwardOnly-diagnostic
PW1-piecewise2-streamingGrad
ResetV7-fused-grouped-g16, if implemented
ResetV7-memory-preserving-vectorized-g16, if implemented
ResetV7-lowrank-r2-fused, if implemented
ResetV7-piecewise-residual-only, if implemented
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
v613_memory_ratio_mean
v614_memory_ratio_mean
v613_step_ratio_mean
v614_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
is_dwm2_patch
dwm2_patch_allowed
```

### 5.4 判断标准

Reproduction pass:

$$
|r_{\text{mem,v614}}-r_{\text{mem,v613}}|\leq0.05.
$$

$$
|r_{\text{step,v614}}-r_{\text{step,v613}}|\leq0.15.
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

## 6. P1：reset-v6 runtime attribution with kernel/op counters

### 6.1 目的

P1 要解决 v6.13 最大诊断缺口：runtime counter 不足。没有 counter，就不能判断 G2 / GR3 / L6 / PW0 为什么失败。

### 6.2 测量对象

```text
G2-grouped-g16-poly1
GR3-g16-vectorized-forward-backward
GR4-g16-batched-group-gemm
L6-lowrank-r2-fast
L4-lowrank-no-intermediate
PW0-piecewise2-forwardOnly-diagnostic
PW1-piecewise2-streamingGrad
DWM2-current-frozen-baseline
MLP-manual-linear-reference
```

### 6.3 Component phases

```text
phase_residual_transform
phase_group_loop_forward
phase_group_loop_backward
phase_vectorized_group_forward
phase_vectorized_group_backward
phase_batched_group_gemm
phase_lowrank_factor_V
phase_lowrank_factor_U
phase_piecewise_bin_index
phase_piecewise_residual_eval
phase_piecewise_knot_grad
phase_mixing_forward
phase_backward_mixing_delta
phase_backward_residual_dx
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

不能填假值。若低层 counter 不可用，至少必须提供：

```text
python_loop_count
torch_op_count
cuda_sync_count
component_time_ms
```

### 6.5 判断标准

Runtime attribution pass 要求至少满足：

```text
component_time_ms available for all core components
and at least two of:
  component_kernel_count available
  component_alloc_count available
  torch_op_count available
  python_loop_count available
  cuda_sync_count available
```

并且必须识别 primary runtime blocker：

```text
primary_runtime_blocker in {
  group_loop_fragmentation,
  vectorized_group_materialization,
  lowrank_factor_materialization,
  chunk_loop_overhead,
  piecewise_residual_eval,
  piecewise_knot_grad,
  mixing_workspace,
  optimizer_update,
  unknown
}
```

如果：

```text
primary_runtime_blocker = unknown
```

则 P2-P5 可继续作为 implementation diagnostic，但 P11 route 必须标记 `RuntimeAttributionIncomplete`。

### 6.6 可视化

```text
p1_component_runtime_waterfall.svg
p1_component_memory_waterfall.svg
p1_component_kernel_count_bar.svg
p1_component_op_count_bar.svg
p1_group_loop_scaling_plot.svg
p1_vectorized_materialization_bar.svg
p1_runtime_blocker_heatmap.svg
```

---

## 7. P2：memory-preserving vectorized grouped fusion

### 7.1 目的

P2 是 v6.14 最核心实验：尝试把 `G2` 的 memory 优势和 `GR3/GR4` 的 runtime 优势合并。

### 7.2 Candidate packages

```text
VG0-G2-grouped-g16-loop-baseline
VG1-GR3-vectorized-g16-baseline
VG2-memory-preserving-vectorized-g16:
  vectorized grouped without materializing full grouped activation tensor

VG3-tiled-vectorized-g16:
  tiled vectorized grouped computation with bounded tile workspace

VG4-fused-grouped-forward-g16:
  fuse residual transform + grouped mixing forward

VG5-fused-grouped-backward-g16:
  fuse grouped mixing delta + residual dx

VG6-fused-grouped-forward-backward-g16:
  combine VG4 and VG5

VG7-batched-group-gemm-no-materialize:
  batched group GEMM without large intermediate materialization

VG8-single-kernel-grouped-mix-g16:
  Triton/custom grouped mixing kernel, if implemented

VG9-grouped-g16-lowrank-crossgroup-light:
  preserve grouped memory, add small cross-group correction
```

### 7.3 必须记录字段

```text
package
group_count
implementation_status
materializes_grouped_activation
materializes_grouped_grad
tile_size
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_improvement_vs_G2
step_improvement_vs_G2
memory_improvement_vs_GR3
step_improvement_vs_GR3
group_workspace_MB
group_materialization_MB
batched_group_gemm_count
group_kernel_count
group_allocation_count
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 7.4 判断标准

P2 useful pass relative to G2:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,G2}}}\leq0.30.
$$

Memory preservation relative to G2:

$$
\frac{M_{\text{new}}}{M_{\text{G2}}}\leq1.05.
$$

Near-pass:

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

If candidate achieves step $\leq1.50$ but memory $>1.05$, it is runtime-repaired but memory-regressed, not survivor.

### 7.5 可视化

```text
p2_grouped_memory_time_pareto.svg
p2_grouped_materialization_bar.svg
p2_g2_vs_gr3_vs_new_waterfall.svg
p2_tiled_grouped_sweep_heatmap.svg
```

---

## 8. P3：grouped custom/batched kernel diagnostic

### 8.1 目的

如果 P2 的 vectorized Torch path 仍不能 preserve memory，P3 测试 custom grouped kernel 是否可以同时控制 memory 和 runtime。

### 8.2 Candidate packages

```text
GK0-grouped-loop-baseline
GK1-batched-group-gemm
GK2-triton-grouped-forward
GK3-triton-grouped-backward
GK4-triton-grouped-forward-backward
GK5-cuda-grouped-forward-backward, optional
GK6-blockdiag-grouped-mix
GK7-grouped-g8-custom
GK8-grouped-g16-custom
```

### 8.3 必须记录字段

```text
package
backend
group_count
block_size
num_warps
tile_size
shared_memory_bytes
registers_per_thread
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
global_load_bytes
global_store_bytes
l2_read_transactions
l2_write_transactions
stall_long_scoreboard
achieved_occupancy
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 8.4 判断标准

Custom grouped kernel pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

or at least relative to G2:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,G2}}}\leq0.30,
$$

while:

$$
\frac{M_{\text{new}}}{M_{\text{G2}}}\leq1.05.
$$

If custom grouped cannot beat GR3/GR4 on memory or time, grouped branch should be stopped.

### 8.5 可视化

```text
p3_custom_grouped_kernel_pareto.svg
p3_backend_comparison_bar.svg
p3_memory_traffic_bar.svg
p3_group_count_custom_heatmap.svg
```

---

## 9. P4：low-rank r2 runtime closure

### 9.1 目的

P4 保留 low-rank r2 作为最平衡 fallback。它没有 grouped 的极端 time failure，但 memory 和 step 都差一点。P4 要判断它是否可以通过 fused low-rank runtime closure 达到 S2。

### 9.2 Candidate packages

```text
LR0-L6-lowrank-r2-current
LR1-r2-fused-forward-v2
LR2-r2-fused-backward-v2
LR3-r2-fused-forward-backward-v2
LR4-r2-no-intermediate-v2
LR5-r2-onebuffer-v2
LR6-r2-vectorized-lowrank-v2
LR7-r2-triton-lowrank-forward
LR8-r2-triton-lowrank-backward
LR9-r2-full-fused-lowrank
LR10-r2-lowrank+tiny-grouped-correction
```

### 9.3 必须记录字段

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
lowrank_materialization_MB
lowrank_kernel_count
lowrank_allocation_count
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 9.4 判断标准

Low-rank useful:

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

### 9.5 可视化

```text
p4_lowrank_package_pareto.svg
p4_lowrank_intermediate_memory_bar.svg
p4_rank_runtime_heatmap.svg
p4_lowrank_step_waterfall.svg
```

---

## 10. P5：piecewise residual-vs-mixing decomposition

### 10.1 目的

P5 不再只测 full piecewise package，而是拆分 piecewise residual 和 mixing，判断 piecewise 失败到底来自 local residual 还是 mixing。

### 10.2 Candidate packages

```text
PW0-piecewise2-forwardOnly-full-current
PW1-piecewise2-residualOnly-forward
PW2-piecewise2-residualOnly-forward-backward
PW3-piecewise2-streamingGrad-noMix
PW4-piecewise2-streamingGrad-lowrankMix-r2
PW5-piecewise2-streamingGrad-groupedMix-vectorized
PW6-piecewise2-streamingGrad-custom-knot
PW7-piecewise2-no-knot-materialization-v2
```

### 10.3 必须记录字段

```text
package
num_bins
active_bins_per_sample
mixing_type
dense_basis_tensor_used
knot_materialized
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
residual_only_memory_ratio
residual_only_step_ratio
mixing_workspace_MB
knot_workspace_MB
knot_grad_workspace_MB
bin_occupancy_entropy
dead_bin_fraction
out_of_grid_fraction
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 10.4 判断标准

Structural pass:

```text
dense_basis_tensor_used = 0
active_bins_per_sample <= 4
manual_backward_available = 1
```

Piecewise residual-only useful:

$$
r_{\text{mem,residual-only}}\leq1.05,
$$

$$
r_{\text{step,residual-only}}\leq1.50.
$$

Full piecewise near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Stability:

$$
\operatorname{out\_of\_grid\_fraction}\leq0.05,
$$

$$
\operatorname{dead\_bin\_fraction}\leq0.30.
$$

If residual-only passes but full fails, mixing is blocker. If residual-only fails, piecewise local is not promising at current implementation level.

### 10.5 可视化

```text
p5_piecewise_residual_vs_full_pareto.svg
p5_knot_workspace_bar.svg
p5_bin_occupancy_heatmap.svg
p5_dead_bin_fraction_bar.svg
p5_piecewise_mixing_blocker_waterfall.svg
```

---

## 11. P6：reset-v7 package selection

### 11.1 目的

P6 汇总 P2-P5，决定是否存在 S0/S1/S2 survivor。

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
  runtime attribution incomplete

S8:
  no viable reset-v7
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
runtime_attribution_pass
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
survivor_type in {S3, S4, S5, S6, S7, S8}
```

### 11.5 可视化

```text
p6_reset_v7_scorecard.svg
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
  Best-reset-v7 + ManualAdanLite
  Best-reset-v7 + ManualAdamW
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

P9 只在 P8 official task pass 后打开。它不是 v6.14 主线。

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
R1-ResetV7Solved:
  reset-v7 gets S0/S1 and P7 pass.
  Open task.

R2-ResetV7NearPass:
  reset-v7 gets S2.
  Continue reset engineering, diagnostic task only.

R3-GroupedMemoryPreservingFusion:
  grouped candidate preserves G2 memory and fixes runtime.
  Continue grouped mainline.

R4-GroupedMemoryRuntimeTradeoff:
  grouped candidate either memory-good/runtime-bad or runtime-good/memory-bad.
  Need custom kernel or stop grouped branch.

R5-LowRankBalancedCandidate:
  low-rank candidate gets S2 or close.
  Continue low-rank mainline.

R6-PiecewiseLocalCandidate:
  piecewise local gets S2 or close.
  Continue piecewise mainline.

R7-RuntimeAttributionIncomplete:
  counters still unavailable.
  Improve profiler before route claim.

R8-NoViableResetV7:
  no reset-v7 candidate gets S2.
  Need new primitive family.

R9-TaskReentryPass:
  memory/time survivor passes limited task.

R10-TaskReentryFail:
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
F5_grouped_materialization_fail
F6_grouped_runtime_fail
F7_lowrank_runtime_fail
F8_piecewise_memory_fail
F9_piecewise_step_fail
F10_runtime_counter_unavailable
F11_task_gated
F12_optimizer_gated
F13_functional_gated
F14_fake_or_proxy_violation
F15_artifact_missing
```

### artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_reset_runtime_attribution.csv
p2_memory_preserving_grouped_fusion.csv
p3_custom_grouped_kernel_diagnostic.csv
p4_lowrank_r2_runtime_closure.csv
p5_piecewise_residual_vs_mixing.csv
p6_reset_v7_selection.csv
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
figures/p1_component_op_count_bar.svg
figures/p2_grouped_memory_time_pareto.svg
figures/p2_g2_vs_gr3_vs_new_waterfall.svg
figures/p3_custom_grouped_kernel_pareto.svg
figures/p4_lowrank_package_pareto.svg
figures/p5_piecewise_residual_vs_full_pareto.svg
figures/p6_family_comparison_pareto.svg
figures/p7_loss_before_after.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：Grouped memory-preserving fusion 成功

如果某 grouped candidate 同时满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

则 grouped branch 成为 v6.15 主线。若 P7 通过，可以打开 diagnostic task。

### Case B：Grouped 仍然 memory/time tradeoff

如果 loop G2 memory 好但太慢，vectorized GR3 time 好但 memory 差，且新 fused candidate 仍不能合并两者，则 grouped branch 需要 custom kernel 或停止。

### Case C：Low-rank r2 达到 S2

如果 low-rank r2 达到 S2，低秩 branch 接替主线。下一步做 one-step probe 与 diagnostic task。

### Case D：Piecewise local residual-only 过但 full fails

说明 mixing 是 blocker。下一轮做 piecewise + bounded mixing，而不是放弃 piecewise。

### Case E：Piecewise residual-only 也失败

说明 piecewise local 不适合当前 setting，停止 piecewise 分支。

### Case F：所有 ResetV7 失败

如果没有 S2 candidate，进入 new primitive family design，不再继续 low-rank/grouped/poly1 reset 小修。

---

## 19. 最终建议

v6.14 的一句话策略是：

$$
\boxed{
\text{把 grouped 的 memory 优势和 vectorized 的 runtime 优势合并；同时保留 low-rank 和 piecewise 作为备选主线。}
}
$$

当前最有价值的线索不是 DWM2，而是：

```text
G2 memory ratio = 1.0460
GR3/GR4 step ratio ≈ 1.98
L6 memory ratio = 1.0786, step ratio ≈ 2.01
Piecewise structural/stability pass but efficiency fail
```

v6.14 的关键成败标准是能否出现第一个 ResetV7 S2 survivor。如果仍然没有 S2，就应该停止当前 ResetV6/V7 分支，进入新的 primitive family 设计。
