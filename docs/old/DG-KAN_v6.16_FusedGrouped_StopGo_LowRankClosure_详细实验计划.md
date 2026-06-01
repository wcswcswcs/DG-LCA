# DG-KAN v6.16：Fused Grouped Kernel 终局判定、Low-Rank S2 Closure 与 ResetV8 Stop/Go 详细实验计划

> 本计划基于 v6.15 final real-only run 制定。v6.15 的最终 route 是 `R4-GroupedTradeoffUnresolved`：grouped tile/stream 能把 memory ratio 压到 `1.0374`，但 step ratio 仍约 `9.0878`；no-xg / vectorized / batched grouped 能把 step ratio 降到约 `1.97-1.99`，但 memory ratio 回退到 `1.127-1.175`；low-rank r2 仍保持较平衡状态，memory ratio 约 `1.0786`、step ratio 约 `2.025`；piecewise residual-only/noMix 已经真实测量，但 memory/time 明显失败。因此 v6.16 不再做普通 torch-level tile/chunk/group sweep，而是执行一次真正的 **fused grouped kernel 终局判定**，同时保留 **low-rank r2 S2 closure** 作为 fallback。如果 v6.16 仍没有 S2 survivor，应停止当前 ResetV8 grouped/low-rank/piecewise 分支，进入新的 primitive family 设计。

---

## 0. 实验整体目标

v6.16 的整体目标不是继续扩大候选搜索，也不是提前打开 task / optimizer / functional correction。v6.16 的核心目标是回答三个 Stop/Go 级问题。

第一个问题：

$$
\boxed{
\text{真正 fused grouped kernel 能否同时保住 GS4/GS5 的 memory，并达到 no-xg/vectorized 的 runtime？}
}
$$

第二个问题：

$$
\boxed{
\text{low-rank r2 能否通过最后一次 fused/runtime closure 达到 S2 near-pass？}
}
$$

第三个问题：

$$
\boxed{
\text{如果 grouped 与 low-rank 均不能达到 S2，是否应停止当前 ResetV8 分支？}
}
$$

v6.15 的关键真实结果如下：

```text
GS4-streamed-grouped-tile1:
  memory ratio mean = 1.0374
  step ratio mean   = 9.0878
  backward ratio mean = 12.7687
  survivor type = S3

GS5-streamed-grouped-tile2:
  memory ratio mean = 1.0493
  step ratio mean   = 5.7495
  backward ratio mean = 7.3270
  survivor type = S3

GS7/GS12 tile8:
  memory ratio mean = 1.1031
  step ratio mean   = 2.68-2.69
  backward ratio mean ≈ 2.56
  survivor type = no S2

GS2/GS11 no-xg:
  memory ratio mean = 1.1270
  step ratio mean   = 1.98-1.99
  backward ratio mean = 1.50-1.51
  survivor type = no S2

GK1 batched / GK7-GK8 no-xg:
  memory ratio mean = 1.1270-1.1749
  step ratio mean   = 1.9679-1.9885
  survivor type = no S2

L6-lowrank-r2-fast:
  memory ratio mean = 1.0786
  step ratio mean   ≈ 2.025
  backward ratio mean ≈ 1.249
  survivor type = no S2

PW2/PW3 residual-only/noMix:
  memory ratio mean ≈ 1.4832
  step ratio mean   ≈ 3.51
  survivor type = no S2
```

v6.16 的最低工程成功标准仍为 S2：

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

正式 task re-entry 标准仍为 S0/S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

强目标为：

$$
r_{\text{mem}}\leq0.90,
$$

$$
r_{\text{step}}\leq1.20.
$$

终极目标仍为：

$$
r_{\text{mem}}\leq0.80.
$$

v6.16 不以 $0.80$ 作为本轮唯一 gate。本轮必须给出清楚路线结论：继续 grouped、转 low-rank、或停止当前 ResetV8 family。

---

## 1. 当前实验进展与判断

### 1.1 已经确认的事实

v6.15 已经确认以下事实：

```text
no fake / no proxy
DWM2 frozen
ResetV8 measured
strict PureKAN candidates nonKAN = 0
manual backward available
gradient correctness pass
residual effect pass
task / optimizer / functional 正确 gate
```

因此 v6.16 不再重新证明这些基础项。它们仍需记录为 contract，但不是本轮主要问题。

### 1.2 当前最强 memory 信号

当前最强 memory 候选是 grouped tile/stream：

```text
GS4 tile1:
  memory ratio mean = 1.0374

GS5 tile2:
  memory ratio mean = 1.0493

G2 / GK0 grouped loop:
  memory ratio mean = 1.0460
```

这些候选已经达到或接近 S2 memory gate：

$$
r_{\text{mem}}\leq1.05.
$$

但它们 runtime 极差，尤其 GS4 backward ratio mean 达到 `12.7687`。这说明 memory 优势来自极细粒度 streaming/tile loop，但当前实现方式严重碎片化。

### 1.3 当前最强 runtime 信号

当前 runtime 较好的 grouped 候选是 no-xg / vectorized / batched grouped：

```text
GK1 / GK7 / GK8 / GS2 / GS11:
  step ratio mean ≈ 1.97-1.99
```

它们将 grouped loop 的 step ratio 从约 $9-10$ 降到约 $2$，说明 grouped 的 runtime 爆炸不是数学必然，而是实现方式导致。

但这些候选的 memory 回退到：

$$
r_{\text{mem}}\approx1.127\text{ 到 }1.175.
$$

因此它们不是 S2 survivor。

### 1.4 当前核心问题

当前核心问题是：

$$
\boxed{
\text{torch-level grouped 实现只能二选一：memory 好但极慢，或 runtime 较好但 memory 回退。}
}
$$

v6.16 必须验证真正 fused grouped kernel 是否能消除这个二选一。

### 1.5 Piecewise local 的状态

v6.15 已经补上 residual-only/noMix：

```text
PW2/PW3 residual-only/noMix:
  structural pass
  stability pass
  residual pass
  grad pass
  memory/time fail
```

这说明 piecewise local 失败不只是 full mixing 组合的问题，residual-only 本体已经重。除非出现全新 workspace model，否则 piecewise 不应作为 v6.16 主线。

### 1.6 Low-rank r2 的状态

Low-rank r2 仍是最平衡 fallback：

```text
L6:
  memory ratio mean ≈ 1.0786
  step ratio mean   ≈ 2.025
```

它离 S2 还差：

$$
1-\frac{1.05}{1.0786}\approx2.65\%
$$

的 memory 降幅，以及约：

$$
1-\frac{1.50}{2.025}\approx25.93\%
$$

的 step 提速。

它比 grouped 更平衡，但现有 torch variants 没有带来 closure。v6.16 只允许 low-rank 做最后一次 fused / Triton / onebuffer closure。

---

## 2. v6.16 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。任何新 DWM2 patch 都不允许进入本轮结果表，除非作为 frozen baseline reproduction。

第三，不允许再做普通 Python group loop sweep。`GS4/GS5/GK0` 只能作为 memory diagnostic，不允许开 P7/P8。

第四，不允许继续普通 torch-level no-xg / batched grouped 小修，除非它实现了明确的 memory-preserving fused / streamed / no materialization 机制。

第五，不允许把 memory-only candidate 当作 survivor。只有同时满足 $r_{\text{mem}}\leq1.05$ 和 $r_{\text{step}}\leq1.50$ 才是 S2。

第六，不允许把 runtime-only candidate 当作 survivor。step ratio 接近 $1.50$ 但 memory ratio $>1.05$ 仍不能进入 P7。

第七，不允许继续 piecewise full package 搜索，除非先出现 residual-only lightweight signal。v6.15 已经显示 residual-only/noMix 也重。

第八，没有 S0/S1/S2 survivor 时，不允许打开：

```text
one-step probe
task re-entry
optimizer exploration
LightSmooth
functional correction
multi-seed confirm
```

第九，如果低层 counter 不可用，必须至少记录 op/call/allocation proxy。不能把 runtime attribution 写成空。

---

## 3. 核心假设

### H1：Grouped branch 只有真正 fused kernel 才可能合并 memory 与 runtime 优势

v6.15 的 torch-level tile/stream sweep 已经证明存在 memory/time tradeoff。H1 假设：只有 fused grouped kernel 能避免两类失败：

```text
Python/group/tile loop 导致 runtime 爆炸
普通 vectorized grouped materialization 导致 memory 回退
```

H1 成立的标准是某个 fused grouped candidate 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

并且相对 GS4：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GS4}}}\leq0.25,
$$

同时相对 GK8 / no-xg：

$$
\frac{M_{\text{new}}}{M_{\text{GK8}}}\leq0.95.
$$

如果 H1 不成立，grouped branch 应停止作为主线。

---

### H2：Grouped memory 回退来自 materialized grouped activation / gradient / einsum intermediate

H2 假设：no-xg / batched grouped runtime 好但 memory 回退，原因是 materialization。例如：

```text
grouped activation tensor
grouped gradient tensor
batched group GEMM workspace
einsum intermediate
layout conversion
contiguous copy
```

H2 成立标准是 runtime attribution 显示：

$$
\frac{M_{\text{grouped materialization}}}{G_{\text{peak}}}\geq0.30,
$$

其中：

$$
G_{\text{peak}}=M_{\text{peak,candidate}}-M_{\text{peak,MLP}}.
$$

H2 修复标准是 fused / streamed candidate 将 grouped materialization 降低至少 $30\%$：

$$
\frac{M_{\text{materialization,new}}}{M_{\text{materialization,baseline}}}\leq0.70.
$$

---

### H3：Tile1/Tile2 memory 优势来自 bounded materialization，但 runtime 爆炸来自 loop/launch count

H3 假设：GS4/GS5 memory 好，是因为 tile 足够小；runtime 差，是因为 tile loop / kernel launch / backward fragmentation 太多。

H3 成立标准：

```text
tile_loop_count scales with step_time
kernel_launch_proxy_count scales with tile_count
component_call_count scales with tile_count
```

H3 修复标准：

$$
\frac{\text{call_count}_{new}}{\text{call_count}_{GS4}}\leq0.30,
$$

且：

$$
\frac{M_{\text{new}}}{M_{\text{GS4}}}\leq1.05.
$$

---

### H4：Low-rank r2 是合理 fallback，但只有 fused r2 才值得继续

H4 假设：low-rank r2 的 memory/time 都接近，但现有 implementation 没有真正 fused。v6.16 只允许测试：

```text
lowrank r2 onebuffer
lowrank r2 Triton forward/backward
lowrank r2 full fused
```

H4 成立标准：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果本轮 low-rank fused variants 仍然没有 S2，则 low-rank r2 只能保留为 diagnostic，不再作为主线小修。

---

### H5：Piecewise local 不是当前主线，除非 residual-only lightweight 改写出现

v6.15 的 PW2/PW3 residual-only/noMix 已经失败：

```text
memory ratio ≈ 1.4832
step ratio ≈ 3.51
```

H5 假设：当前 piecewise local implementation 本体太重，不应继续围绕 full package 修。只有新的 residual-only implementation 能满足：

$$
r_{\text{mem,residual-only}}\leq1.10,
$$

$$
r_{\text{step,residual-only}}\leq1.75,
$$

才允许 piecewise branch 继续。

否则 piecewise branch 停止。

---

### H6：如果 v6.16 仍无 S2，应停止 ResetV8 当前分支

H6 是本轮 Stop/Go 假设。如果以下均完成：

```text
fused grouped kernel measured or explicitly not implemented
low-rank r2 fused closure measured or explicitly not implemented
piecewise residual-only diagnostic measured
runtime attribution proxy available
```

仍没有 S2 survivor，则 route 应输出：

```text
R8-NoViableResetV8
```

下一轮必须进入新的 primitive family design，而不是继续 grouped/lowrank/piecewise 小修。

---

## 4. 实验阶段总览

v6.16 分为十二个阶段：

```text
P0: v6.15 reproduction and contract check
P1: grouped materialization / loop attribution closure
P2: fused grouped forward/backward kernel
P3: grouped fused tile tuning and full-step integration
P4: low-rank r2 final fused closure
P5: piecewise residual-only lightweight recheck
P6: reset-v9 package selection
P7: one-step loss and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P6 是核心。P7 只对 S0/S1/S2 candidate 运行。P8-P10 默认关闭。

---

## 5. P0：v6.15 reproduction and contract check

### 5.1 目的

确认 v6.16 与 v6.15 baseline 可比，确保 DWM2 继续冻结，ResetV9 新候选仍然是 strict PureKAN / graph-free。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-frozen-baseline

GS4-streamed-grouped-tile1
GS5-streamed-grouped-tile2
GS2-no-xg-grouped
GK8-grouped-g16-custom-no-xg
L6-lowrank-r2-fast
PW2-piecewise2-residualOnly-forward-backward
PW3-piecewise2-streamingGrad-noMix

ResetV9-fused-grouped-forward, if implemented
ResetV9-fused-grouped-backward, if implemented
ResetV9-fused-grouped-forward-backward, if implemented
ResetV9-lowrank-r2-full-fused, if implemented
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
v615_memory_ratio_mean
v616_memory_ratio_mean
v615_step_ratio_mean
v616_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
is_dwm2_patch
dwm2_patch_allowed
```

### 5.4 判断标准

Reproduction pass:

$$
|r_{\text{mem,v616}}-r_{\text{mem,v615}}|\leq0.05.
$$

$$
|r_{\text{step,v616}}-r_{\text{step,v615}}|\leq0.15.
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

## 6. P1：grouped materialization / loop attribution closure

### 6.1 目的

P1 要把 grouped memory/runtime tradeoff 拆清楚。v6.15 已有 code-path proxy counts，但 v6.16 必须更明确地区分：

```text
loop runtime
kernel/call fragmentation
materialized grouped activation
materialized grouped grad
einsum/batched GEMM workspace
tile workspace
layout/contiguous copy
```

### 6.2 测量对象

```text
GS4-streamed-grouped-tile1
GS5-streamed-grouped-tile2
GS2-no-xg-grouped
GK8-no-xg-grouped-g16
VG3-tile4
L6-lowrank-r2-fast
DWM2-current-frozen-baseline
MLP-manual-linear-reference
```

### 6.3 Component phases

```text
phase_grouped_residual_transform
phase_grouped_mixing_forward
phase_grouped_tile_loop_forward
phase_grouped_tile_loop_backward
phase_grouped_no_xg_forward
phase_grouped_no_xg_backward
phase_grouped_materialization
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
component_fraction_of_step_time
component_fraction_of_peak_gap

python_loop_count
torch_op_count
component_call_count
cuda_event_count
cuda_sync_count
allocation_proxy_count
kernel_launch_proxy_count

materialized_group_activation_MB
materialized_group_grad_MB
materialized_einsum_intermediate_MB
tile_workspace_MB
layout_conversion_MB
contiguous_copy_MB

tile_count
group_count
tile_size

grad_relerr_max
grad_cos_min
```

如果 low-level Nsight counter 可用，额外记录：

```text
global_load_bytes
global_store_bytes
l2_read_transactions
l2_write_transactions
stall_long_scoreboard
achieved_occupancy
```

如果不可用，写：

```text
metric_unavailable
```

但以下 proxy 不能全缺：

```text
component_time_ms
python_loop_count
torch_op_count
component_call_count
materialized_group_activation_MB
tile_workspace_MB
```

### 6.5 判断标准

Runtime attribution pass:

```text
component_time_ms available for all core components
and at least three of:
  python_loop_count
  torch_op_count
  component_call_count
  allocation_proxy_count
  kernel_launch_proxy_count
  materialized_group_activation_MB
  tile_workspace_MB
available
```

Grouped materialization blocker pass:

$$
\frac{M_{\text{grouped materialization}}}{G_{\text{peak}}}\geq0.30.
$$

Group-loop blocker pass:

$$
\frac{T_{\text{group loop}}}{T_{\text{step}}}\geq0.50.
$$

如果 blocker 仍 unknown，P11 route 不能 claim grouped 分支被证伪，只能 claim attribution incomplete。

### 6.6 可视化

```text
p1_grouped_runtime_waterfall.svg
p1_grouped_materialization_bar.svg
p1_loop_count_vs_step_time.svg
p1_tile_workspace_vs_memory.svg
p1_grouped_blocker_heatmap.svg
```

---

## 7. P2：fused grouped forward/backward kernel

### 7.1 目的

P2 是 v6.16 的核心。它必须测试真正 fused grouped kernel，而不是普通 torch loop / no-xg / tile sweep。

### 7.2 Candidate packages

```text
FG0-GS4-tile1-memory-reference
FG1-GS2-no-xg-runtime-reference
FG2-GK8-no-xg-runtime-reference

FG3-fused-grouped-forward-g16:
  fused residual transform + grouped mixing forward

FG4-fused-grouped-backward-g16:
  fused grouped mixing delta + residual dx

FG5-fused-grouped-forward-backward-g16:
  combine FG3 + FG4

FG6-fused-grouped-forward-backward-no-materialize:
  avoid materialized grouped activation / grouped grad

FG7-tiled-fused-grouped-tile2:
  tile2 memory path with fused kernel

FG8-tiled-fused-grouped-tile4:
  tile4 compromise path with fused kernel

FG9-single-kernel-grouped-mix:
  Triton/CUDA single grouped mix kernel, if implemented

FG10-blockdiag-grouped-mix:
  block-diagonal implementation if available
```

### 7.3 必须记录字段

```text
package
backend
implementation_status
group_count
tile_size
fused_forward
fused_backward
materializes_group_activation
materializes_group_grad
materializes_einsum_intermediate

memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean

memory_improvement_vs_GS4
step_improvement_vs_GS4
memory_improvement_vs_GK8
step_improvement_vs_GK8

component_call_count
kernel_launch_proxy_count
group_workspace_MB
tile_workspace_MB
materialized_group_activation_MB
materialized_group_grad_MB

grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 7.4 判断标准

Grouped fused useful pass:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GS4}}}\leq0.25.
$$

Memory preservation pass:

$$
\frac{M_{\text{new}}}{M_{\text{GS4}}}\leq1.05.
$$

S2 near-pass:

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

If fused kernel is not implemented, P2 must explicitly write:

```text
not_implemented_fused_kernel_absent
```

and P11 must decide whether grouped branch stops or waits for custom kernel implementation.

### 7.5 可视化

```text
p2_fused_grouped_pareto.svg
p2_fused_vs_loop_vs_noxg_waterfall.svg
p2_materialization_reduction_bar.svg
p2_tile_fused_sweep_heatmap.svg
```

---

## 8. P3：grouped fused tile tuning and full-step integration

### 8.1 目的

如果 P2 出现 any fused grouped candidate with partial improvement，P3 做 full-step tuning。P3 不再测试未 fused 的 GS4/GS5 sweep，只测 fused variants。

### 8.2 Candidate packages

```text
GT0-best-P2-fused-grouped
GT1-best-fused-tile1
GT2-best-fused-tile2
GT3-best-fused-tile4
GT4-best-fused-no-materialize
GT5-best-fused-blockdiag
GT6-best-fused-with-update-prep
```

### 8.3 必须记录字段

```text
package
base_package
tile_size
update_prep_fused
blockdiag_enabled
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
forward_ratio_mean
shape_stability_score
batch_scaling_slope_memory
batch_scaling_slope_step
grad_relerr_max
grad_cos_min
residual_effect_pass
```

Shape stability:

$$
S_{\text{shape}}=1-\frac{\operatorname{std}(r_{\text{mem}})}{\operatorname{mean}(r_{\text{mem}})+\epsilon}.
$$

### 8.4 判断标准

P3 S2:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

and:

$$
S_{\text{shape}}\geq0.85.
$$

P3 S1:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

If P2/P3 both fail and fused implementation exists, grouped branch stops.

### 8.5 可视化

```text
p3_fused_grouped_fullstep_pareto.svg
p3_batch_depth_stability_heatmap.svg
p3_shape_scaling_curve.svg
p3_s0_s1_s2_boundary_plot.svg
```

---

## 9. P4：low-rank r2 final fused closure

### 9.1 目的

P4 是 low-rank 的最后一次 closure。它是 grouped 失败时最现实的 fallback。v6.15 后若 P4 仍没有 S2，则 low-rank r2 不再继续小修。

### 9.2 Candidate packages

```text
LR0-L6-r2-current
LR1-r2-fused-forward-v4
LR2-r2-fused-backward-v4
LR3-r2-fused-forward-backward-v4
LR4-r2-onebuffer-v4
LR5-r2-no-materialize-v4
LR6-r2-triton-lowrank-forward
LR7-r2-triton-lowrank-backward
LR8-r2-full-fused-lowrank
LR9-r1-fast-official-if-residual-pass
LR10-r2-lowrank+micro-grouped-correction
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
torch_op_count
python_loop_count
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 9.4 判断标准

Low-rank useful pass:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75.
$$

Memory no-regression:

$$
\frac{M_{\text{new}}}{M_{\text{L6}}}\leq1.05.
$$

S2:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

If P4 fails, low-rank r2 branch becomes diagnostic only.

### 9.5 可视化

```text
p4_lowrank_final_pareto.svg
p4_lowrank_materialization_bar.svg
p4_lowrank_runtime_waterfall.svg
p4_rank1_rank2_comparison.svg
```

---

## 10. P5：piecewise residual-only lightweight recheck

### 10.1 目的

v6.15 已经显示 PW2/PW3 residual-only/noMix 也重。P5 只允许测试一种新的 lightweight residual-only path。如果仍失败，piecewise 分支停止。

### 10.2 Candidate packages

```text
PW0-piecewise2-residualOnly-current-reference
PW1-piecewise2-residualOnly-lightweight-v2
PW2-piecewise2-residualOnly-no-knot-materialization-v3
PW3-piecewise2-residualOnly-triton-knot, if implemented
PW4-piecewise2-residualOnly-static-grid-diagnostic
```

### 10.3 必须记录字段

```text
package
num_bins
active_bins_per_sample
dense_basis_tensor_used
knot_materialized
streaming_grad_enabled
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
residual_only_memory_ratio
residual_only_step_ratio
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

Piecewise residual-only continuation pass:

$$
r_{\text{mem,residual-only}}\leq1.10,
$$

$$
r_{\text{step,residual-only}}\leq1.75.
$$

S2 requires:

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

If continuation pass fails, piecewise branch stops.

### 10.5 可视化

```text
p5_piecewise_lightweight_pareto.svg
p5_knot_workspace_bar.svg
p5_bin_occupancy_heatmap.svg
p5_dead_bin_fraction_bar.svg
```

---

## 11. P6：reset-v9 package selection

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
  no viable reset-v9
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

Open diagnostic task only if:

```text
survivor_type == S2 and P7 pass
```

No task if:

```text
survivor_type in {S3, S4, S5, S6, S7, S8}
```

### 11.5 可视化

```text
p6_reset_v9_scorecard.svg
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
  Best-reset-v9 + ManualAdanLite
  Best-reset-v9 + ManualAdamW
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

P9 只在 P8 official task pass 后打开。它不是 v6.16 主线。

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
R1-ResetV9Solved:
  reset-v9 gets S0/S1 and P7 pass.
  Open task.

R2-ResetV9NearPass:
  reset-v9 gets S2.
  Continue reset engineering, diagnostic task only.

R3-GroupedFusedSolved:
  fused grouped candidate preserves GS4 memory and fixes runtime.
  Continue grouped mainline.

R4-GroupedCustomKernelNeeded:
  torch-level grouped remains tradeoff, but fused/Triton not implemented.
  Route requires real custom kernel or stop grouped.

R5-GroupedBranchStopped:
  fused/custom grouped implemented and still no S2.
  Stop grouped branch.

R6-LowRankS2Candidate:
  low-rank r2 gets S2.
  Continue low-rank mainline.

R7-LowRankBranchStopped:
  low-rank fused closure fails.
  Stop low-rank branch.

R8-PiecewiseBranchStopped:
  piecewise residual-only continuation pass fails.
  Stop piecewise branch.

R9-NoViableResetV9:
  no branch gets S2.
  Enter new primitive family design.

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
best_forward_ratio
memory_improvement_vs_current
step_improvement_vs_current
survivor_type
residual_effect_pass
grad_pass
runtime_attribution_pass
fused_grouped_measured
fused_grouped_pass
lowrank_closure_pass
piecewise_continuation_pass
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
F7_grouped_fused_not_implemented
F8_grouped_fused_no_effect
F9_lowrank_runtime_fail
F10_lowrank_memory_fail
F11_piecewise_memory_fail
F12_piecewise_step_fail
F13_runtime_counter_unavailable
F14_task_gated
F15_optimizer_gated
F16_functional_gated
F17_fake_or_proxy_violation
F18_artifact_missing
```

### artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_grouped_attribution.csv
p2_fused_grouped_kernel.csv
p3_fused_grouped_fullstep_tuning.csv
p4_lowrank_final_closure.csv
p5_piecewise_lightweight_recheck.csv
p6_reset_v9_selection.csv
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
figures/p1_grouped_runtime_waterfall.svg
figures/p1_grouped_materialization_bar.svg
figures/p2_fused_grouped_pareto.svg
figures/p2_fused_vs_loop_vs_noxg_waterfall.svg
figures/p3_fused_grouped_fullstep_pareto.svg
figures/p4_lowrank_final_pareto.svg
figures/p5_piecewise_lightweight_pareto.svg
figures/p6_family_comparison_pareto.svg
figures/p7_loss_before_after.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：fused grouped kernel 达到 S2

如果 grouped fused candidate 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

则 grouped branch 成为主线。P7 one-step probe 必须打开。

### Case B：fused grouped 未实现

如果 fused grouped 仍为 `not_implemented`，且 torch-level grouped 仍 tradeoff，则 route 必须是：

```text
R4-GroupedCustomKernelNeeded
```

不能继续用 torch loop / no-xg 小修冒充新进展。

### Case C：fused grouped 实现但仍无 S2

如果 fused grouped 已实现且仍无 S2，则 grouped branch 停止。

### Case D：low-rank r2 达到 S2

如果 low-rank r2 达到 S2，则 low-rank branch 接替主线。

### Case E：low-rank r2 final closure 失败

如果 low-rank r2 fused closure 仍未达到 S2，则 low-rank branch 停止小修。

### Case F：piecewise residual-only lightweight 失败

如果 piecewise residual-only lightweight 仍未达到 continuation pass，则 piecewise branch 停止。

### Case G：所有分支均失败

如果 grouped、low-rank、piecewise 都没有 S2 candidate，则进入 new primitive family design。

---

## 19. 最终建议

v6.16 的一句话策略是：

$$
\boxed{
\text{把 grouped 分支推进到真正 fused kernel，或者停止 grouped 分支。}
}
$$

同时用 low-rank r2 做最后一次 S2 closure，用 piecewise residual-only 做最后一次 lightweight recheck。

如果 v6.16 仍然没有 S0/S1/S2 candidate，就不要再继续 ResetV8/ResetV9 小修，而应进入新的 bounded-workspace primitive family 设计。
