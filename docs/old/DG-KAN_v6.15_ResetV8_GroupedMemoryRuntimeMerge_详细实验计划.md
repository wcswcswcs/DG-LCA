# DG-KAN v6.15：Grouped 分支终局判定、Low-Rank S2 Closure 与 Piecewise Residual-Only 诊断详细实验计划

> 本计划基于 v6.14 final real-only run 制定。v6.14 的最终 route 是 `R4-GroupedMemoryRuntimeTradeoff`：grouped loop 分支保住了最强 memory signal，`G2/GK0` 的 memory ratio mean 为 `1.0460`，但 step ratio 仍在 `9.25-9.91`；vectorized/no-xg grouped 分支把 step ratio 降到约 `1.89-2.03`，但 memory ratio 回退到 `1.127-1.175`；low-rank r2 仍是相对平衡 fallback，memory ratio mean 为 `1.0786`、step ratio 约 `2.0`；piecewise local 已实现并通过 structural/stability/residual gates，但 memory/time 明显失败。v6.15 的目标不是继续扩大搜索，而是完成三条分支的 stop/go 判定：**grouped 是否能通过真正 fused/custom kernel 合并 memory 与 runtime 优势；low-rank 是否能压到 S2；piecewise residual-only 是否值得保留。**

---

## 0. 实验整体目标

v6.15 的整体目标是回答下面三个问题。

第一个问题：

$$
\boxed{
\text{Grouped branch 的 memory 优势和 vectorized runtime 优势能否合并到同一个 candidate？}
}
$$

第二个问题：

$$
\boxed{
\text{Low-rank r2 是否能通过小幅 memory repair + runtime closure 达到 S2 near-pass？}
}
$$

第三个问题：

$$
\boxed{
\text{Piecewise local 的失败是否来自 mixing，而不是 residual 本体？}
}
$$

v6.14 的事实基线如下：

```text
G2 / GK0 grouped loop:
  memory ratio mean = 1.0460
  step ratio mean   = 9.25 - 9.91
  backward ratio mean = 7.33 - 7.95
  residual effect pass = true
  grad pass = true
  survivor type = S3, memory-only diagnostic

GR3 / GR4 / GK1 vectorized grouped:
  memory ratio mean = 1.1749
  step ratio mean   = 1.8879 - 2.0074
  backward ratio mean = 1.4223 - 1.5265
  residual effect pass = true
  grad pass = true
  no survivor because memory regresses

VG2 / VG7 no-xg grouped:
  memory ratio mean = 1.1270
  step ratio mean   = 2.0307 - 2.0312
  backward ratio mean = 1.5579 - 1.5615
  residual effect pass = true
  grad pass = true
  still no survivor

VG3 tiled vectorized g16 tile4:
  memory ratio mean = 1.0673
  step ratio mean   = 3.7881
  backward ratio mean = 4.2786
  residual effect pass = true
  grad pass = true
  memory closer but runtime too slow

L6 lowrank r2:
  memory ratio mean = 1.0786
  step ratio mean   = about 2.0
  backward ratio mean = about 1.23 - 1.27
  residual effect pass = true
  grad pass = true
  balanced fallback but no S2

Piecewise local:
  PW0 memory ratio mean = 1.3277
  PW0 step ratio mean   = 3.7215
  PW1 memory ratio mean = 1.4862
  PW1 step ratio mean   = 4.0981
  structural/stability/residual pass = true
  no near-pass
```

v6.14 的最终 route 是：

```text
R4-GroupedMemoryRuntimeTradeoff
primary_blocker = runtime_too_slow
next_required_implementation = fused_grouped_kernel_or_stop_grouped_branch
```

v6.15 的最低工程成功标准是找到至少一个真实 candidate 满足：

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

v6.15 不把 $0.80$ 作为本轮唯一 gate。本轮现实目标是：找到 S2 survivor，或者明确停止 grouped / low-rank / piecewise 当前实现分支。

---

## 1. 当前实验进展与问题判断

### 1.1 已经成立的部分

v6.14 继续保持 real-only 纪律。P0-P10 的 fake/proxy 总和为 0；DWM2 继续 frozen，没有新增 DWM2 patch；所有 measured reset candidates 保持 `nonKAN=0`、manual backward、gradient pass 和 residual effect pass。未实现的 fused grouped/Triton/CUDA/onebuffer/piecewise residual-only 候选明确写为 `not_implemented`，P7-P10 正确 gate。

因此 v6.15 不需要重新证明：

```text
strict PureKAN accounting
manual gradient correctness
residual effect 是否存在
no-fake / no-proxy discipline
```

当前真正问题是：

```text
memory/time 不能同时达标
runtime attribution 仍不够低层
grouped memory/runtime tradeoff 没有合并
low-rank memory/time 各差一点
piecewise residual-only 未拆清楚
```

### 1.2 v6.14 的最大正向信号

v6.14 最大正向信号是：grouped 分支的两个极端已经很清楚。

一端是 grouped loop：

```text
memory ratio ≈ 1.0460
step ratio ≈ 9.25 - 9.91
```

另一端是 vectorized/no-xg grouped：

```text
memory ratio ≈ 1.127 - 1.175
step ratio ≈ 1.89 - 2.03
```

这说明 grouped 的 runtime 爆炸不是数学必然。通过 vectorization / no-xg einsum 可以把 step time 从约 $10\times$ 降到约 $2\times$。但 vectorized 路径会 materialize 额外 grouped workspace，使 memory 回退。因此当前目标不是再证明 grouped 有 memory 或 runtime 信号，而是把这两者合并。

### 1.3 当前主要问题

v6.14 的最大 failure 仍是：

```text
F9_runtime_counter_unavailable = 1980
```

这意味着 P1 虽然有 code-loop/phase-field attribution，但没有真实低层 kernel/allocation counter。因此我们仍不能完全确认：

```text
G2 的 step 爆炸来自多少 Python loop、多少 kernel launch、多少小 GEMM；
GR3/VG2 的 memory 回退来自 grouped activation materialization、einsum intermediate、layout conversion 还是 grad temp；
low-rank r2 的 step ~2.0 来自 lowrank projection、update、mixing 还是 allocation；
piecewise local 的 memory/time 失败来自 residual/knot，还是 mixing。
```

v6.15 的第一任务就是补齐 enough-to-decide 的 runtime attribution。即使 Nsight 仍不可用，也必须有稳定的 `torch_op_count`、`python_loop_count`、`cuda_event_count`、`kernel_launch_proxy_count`、`allocation_count` 或自定义 instrumentation 代理，不能继续只写 `metric_unavailable`。

---

## 2. v6.15 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。v6.15 不允许新增 DWM2 局部 patch。

第三，不允许把 memory-only candidate 作为 survivor。`G2/GK0` 只有在 step ratio 降到 $1.50$ 以下时，才允许进入 P7 one-step probe。

第四，不允许把 vectorized grouped 的 runtime improvement 写成成功，除非 memory ratio 同时回到 $1.05$ 以下。

第五，不允许继续只做 phase-level attribution。v6.15 必须至少取得可靠的 loop/op/allocation 代理计数；如果低层 counter 不可用，route 必须显式标记 `RuntimeAttributionIncomplete`。

第六，不允许在没有 S0/S1/S2 survivor 前打开：

```text
task re-entry
optimizer exploration
LightSmooth
functional correction
multi-seed confirm
```

第七，不允许继续把 piecewise full package 失败直接等同于 piecewise residual 失败。必须先测 residual-only / noMix diagnostic。

第八，不允许只做更多 rank/group/chunk 超参，而不做实现路径诊断。所有 sweep 必须绑定 runtime attribution。

---

## 3. 核心假设

### H1：Grouped memory/runtime tradeoff 的根因是 materialization 方式，而不是 grouped primitive 本身

v6.14 中：

```text
G2:
  memory ≈ 1.0460, step ≈ 9.9

GR3/GK1/VG2:
  step ≈ 1.9-2.0, memory ≈ 1.127-1.175
```

H1 假设：loop path 省 memory 是因为它避免了大 grouped tensor materialization；vectorized path 快是因为减少 loop/kernel，但 materialize 了 grouped workspace。要成功，必须实现 streaming/tiled/fused grouped kernel，使其同时满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

H1 成立的诊断标准是：

$$
\frac{M_{\text{grouped materialization}}}{G_{\text{peak}}}\geq0.30
$$

for vectorized path, and:

$$
\frac{T_{\text{group loop}}}{T_{\text{step}}}\geq0.50
$$

for loop path.

其中：

$$
G_{\text{peak}}=M_{\text{peak,candidate}}-M_{\text{peak,MLP}}.
$$

### H2：Memory-preserving vectorized grouped 需要 tile/stream，不是普通 no-xg einsum

`VG2` 和 `VG7` 已经把 memory 从 `1.1749` 降到 `1.1270`，但仍高于 $1.05$；`VG3` 把 memory 进一步降到 `1.0673`，但 step 升到 `3.79`。H2 假设：需要更好的 tile/stream 策略，避免 tile 太小导致 runtime 爆炸，避免 tile 太大导致 memory 回退。

H2 的目标是找到 tile size $t^\*$，使得：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

若所有 tile 都表现为：

```text
small tile: memory good, step bad
large tile: step better, memory bad
```

则 grouped 分支需要 Triton/CUDA custom kernel，不能继续 Torch-level implementation。

### H3：Low-rank r2 是最现实的 fallback，但需要把 step 降约 25%、memory 降约 3%

`L6-lowrank-r2-fast` 的状态是：

$$
r_{\text{mem}}=1.0786,
$$

$$
r_{\text{step}}\approx2.0.
$$

到 S2 需要：

$$
1-\frac{1.05}{1.0786}\approx2.65\%
$$

memory 降幅，以及：

$$
1-\frac{1.50}{2.0}\approx25\%
$$

step 提速。H3 假设：如果 low-rank 的主要耗时来自 factor projection / intermediate materialization，则 fused r2 low-rank path 可以进入 S2。

H3 成立标准是某个 low-rank candidate 满足：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75,
$$

并且：

$$
\frac{M_{\text{new}}}{M_{\text{L6}}}\leq1.05.
$$

### H4：Piecewise local 失败可能来自 mixing，不一定来自 residual/knot

v6.14 的 measured piecewise full packages 都很差，但 residual-only/noMix 仍未实现。H4 假设：piecewise residual 本身可能轻，但和 mixing 组合后爆 memory/time。

H4 成立标准：

```text
piecewise residual-only candidate satisfies r_mem <= 1.05 and r_step <= 1.50
full piecewise candidate fails
```

如果 residual-only 也 fails，则 piecewise local 不适合当前 setting。

### H5：Runtime counter closure 是路线判定必要条件

v6.14 的最大 failure 是 `F9_runtime_counter_unavailable=1980`。H5 假设：如果继续没有 kernel/op/allocation counter，就无法判断分支是否真正不可修。

v6.15 P1 必须至少获得：

```text
python_loop_count
torch_op_count
component_time_ms
component_allocation_count or allocation_proxy_count
kernel_launch_proxy_count or component_kernel_count
```

如果仍全部不可用，route 应为：

```text
R7-RuntimeAttributionIncomplete
```

而不是 claim grouped / low-rank / piecewise 已经被证伪。

### H6：如果 v6.15 仍无 S2 survivor，应停止 ResetV7 当前分支

H6 是 Stop/Go 假设。如果完成：

```text
grouped fused/tiled/streamed implementation
low-rank r2 runtime closure
piecewise residual-only diagnostic
runtime attribution
```

仍没有 S2 survivor，则应输出：

```text
R8-NoViableResetV8
```

下一轮进入新的 primitive family design，而不是继续 reset-v7 小修。

---

## 4. 实验阶段总览

v6.15 分为十二个阶段：

```text
P0: v6.14 reproduction and contract check
P1: runtime attribution closure with op/allocation proxies
P2: memory-preserving grouped tile/stream sweep
P3: fused grouped custom-kernel go/no-go
P4: low-rank r2 S2 closure
P5: piecewise residual-only / mixing decomposition
P6: reset-v8 package selection
P7: one-step loss and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P6 是核心。P7 只对 S0/S1/S2 candidate 运行。P8-P10 默认关闭。

---

## 5. P0：v6.14 reproduction and contract check

### 5.1 目的

确认 v6.15 与 v6.14 baseline 可比，并确保 DWM2 继续冻结、ResetV8 新候选是 strict PureKAN / graph-free。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-frozen-baseline
G2-grouped-g16-poly1
GR3-g16-vectorized-forward-backward
VG2-memory-preserving-vectorized-g16
VG3-tiled-vectorized-g16-tile4
GK8-grouped-g16-custom-no-xg
L6-lowrank-r2-fast
PW0-piecewise2-forwardOnly-full-current
PW1-piecewise2-streamingGrad
ResetV8-grouped-stream-tile, if implemented
ResetV8-lowrank-r2-fused, if implemented
ResetV8-piecewise-residual-only, if implemented
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
v614_memory_ratio_mean
v615_memory_ratio_mean
v614_step_ratio_mean
v615_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
is_dwm2_patch
dwm2_patch_allowed
```

### 5.4 判断标准

Reproduction pass:

$$
|r_{\text{mem,v615}}-r_{\text{mem,v614}}|\leq0.05.
$$

$$
|r_{\text{step,v615}}-r_{\text{step,v614}}|\leq0.15.
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

## 6. P1：runtime attribution closure with op/allocation proxies

### 6.1 目的

P1 解决 v6.14 的最大诊断缺口：`F9_runtime_counter_unavailable=1980`。本阶段要求即使拿不到 Nsight，也要用 wrapper instrumentation 和 profiler proxy 取得足够可判定的 runtime attribution。

### 6.2 测量对象

```text
G2-grouped-g16-poly1
GR3-g16-vectorized-forward-backward
VG2-memory-preserving-vectorized-g16
VG3-tiled-vectorized-g16-tile4
GK8-grouped-g16-custom-no-xg
L6-lowrank-r2-fast
PW0-piecewise2-forwardOnly-full-current
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
phase_tiled_group_forward
phase_tiled_group_backward
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
component_fraction_of_step_time
component_fraction_of_peak_gap

python_loop_count
torch_op_count
cuda_event_count
cuda_sync_count
allocation_proxy_count
kernel_launch_proxy_count
component_call_count

component_alloc_count
component_kernel_count
component_gemm_count
component_elementwise_count
component_custom_kernel_count

group_loop_count
chunk_loop_count
tile_count
materialized_group_tensor_MB
materialized_lowrank_tensor_MB
materialized_piecewise_tensor_MB

global_load_bytes
global_store_bytes
l2_read_transactions
l2_write_transactions
stall_long_scoreboard
metric_availability

grad_relerr_max
grad_cos_min
```

如果低层 counter 不可用，必须写：

```text
metric_unavailable
```

但以下 proxy 不允许全部缺失：

```text
python_loop_count
torch_op_count
component_call_count
component_time_ms
```

### 6.5 判断标准

Runtime attribution pass 要求：

```text
component_time_ms available for all core components
and at least three of:
  python_loop_count available
  torch_op_count available
  component_call_count available
  allocation_proxy_count available
  kernel_launch_proxy_count available
  component_kernel_count available
  component_alloc_count available
```

并且必须识别 primary blocker：

```text
primary_runtime_blocker in {
  group_loop_fragmentation,
  vectorized_group_materialization,
  tiled_group_loop_overhead,
  lowrank_factor_materialization,
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

则 P2-P5 仍可做 diagnostic，但 P11 route 必须标记 `RuntimeAttributionIncomplete`。

### 6.6 可视化

```text
p1_component_runtime_waterfall.svg
p1_component_op_count_bar.svg
p1_component_call_count_bar.svg
p1_group_loop_scaling_plot.svg
p1_materialization_memory_bar.svg
p1_runtime_blocker_heatmap.svg
```

---

## 7. P2：memory-preserving grouped tile/stream sweep

### 7.1 目的

P2 是 v6.15 grouped 分支的核心实验。目标是找到介于 G2 loop 与 GR3 vectorized 之间的 tile/stream point：保住 memory，同时显著降低 step。

### 7.2 Candidate packages

```text
GS0-G2-loop-baseline
GS1-GR3-vectorized-baseline
GS2-VG2-no-xg-baseline
GS3-VG3-tile4-baseline

GS4-streamed-grouped-tile1
GS5-streamed-grouped-tile2
GS6-streamed-grouped-tile4
GS7-streamed-grouped-tile8
GS8-streamed-grouped-tile16

GS9-two-level-grouped-tile4x4
GS10-two-level-grouped-tile8x2

GS11-memory-preserving-vectorized-no-materialize-v2
GS12-tiled-no-xg-g16-tile8
GS13-tiled-no-xg-g16-tile16
```

### 7.3 必须记录字段

```text
package
group_count
tile_size
streaming_enabled
two_level_tile_enabled
materializes_grouped_activation
materializes_grouped_grad
implementation_status

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
tile_workspace_MB
tile_loop_count
group_kernel_count
group_allocation_count
torch_op_count
python_loop_count

grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 7.4 判断标准

Memory preservation relative to G2:

$$
\frac{M_{\text{new}}}{M_{\text{G2}}}\leq1.05.
$$

Runtime repair relative to G2:

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

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

If candidate satisfies runtime but not memory, classify:

```text
runtime-repaired-memory-regressed
```

If candidate satisfies memory but not runtime, classify:

```text
memory-preserved-runtime-fail
```

Only candidates satisfying both are S2.

### 7.5 可视化

```text
p2_grouped_tile_stream_pareto.svg
p2_tile_size_memory_curve.svg
p2_tile_size_step_curve.svg
p2_materialization_vs_memory.svg
p2_g2_gr3_new_waterfall.svg
```

---

## 8. P3：fused grouped custom-kernel go/no-go

### 8.1 目的

如果 P2 仍然是 memory/runtime tradeoff，P3 做 grouped branch 的最后一次 kernel go/no-go：Triton/custom grouped mix 是否能同时保 memory 和修 runtime。

### 8.2 Candidate packages

```text
FG0-G2-loop-reference
FG1-GR3-vectorized-reference
FG2-triton-grouped-forward-g16
FG3-triton-grouped-backward-g16
FG4-triton-grouped-forward-backward-g16
FG5-triton-grouped-no-materialize-g16
FG6-cuda-grouped-forward-backward-g16, optional
FG7-blockdiag-grouped-mix-g16
FG8-fused-grouped-residual-mix-g16
```

### 8.3 必须记录字段

```text
package
backend
group_count
block_size
tile_size
num_warps
shared_memory_bytes
registers_per_thread
materializes_grouped_tensor

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

Custom grouped pass:

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

and:

$$
\frac{M_{\text{new}}}{M_{\text{G2}}}\leq1.05.
$$

If custom grouped cannot beat both:

```text
G2 memory
GR3 runtime
```

then grouped branch must be stopped as mainline.

### 8.5 可视化

```text
p3_custom_grouped_kernel_pareto.svg
p3_backend_comparison_bar.svg
p3_memory_traffic_bar.svg
p3_grouped_go_nogo_dashboard.svg
```

---

## 9. P4：low-rank r2 S2 closure

### 9.1 目的

Low-rank r2 是最平衡 fallback。P4 目标是把 L6 从 memory `1.0786`、step `~2.0` 推到 S2。

### 9.2 Candidate packages

```text
LR0-L6-r2-current
LR1-r2-fused-forward-v3
LR2-r2-fused-backward-v3
LR3-r2-fused-forward-backward-v3
LR4-r2-onebuffer-v3
LR5-r2-no-materialize-v3
LR6-r2-triton-lowrank-forward
LR7-r2-triton-lowrank-backward
LR8-r2-full-fused-lowrank
LR9-r1-fast-diagnostic
LR10-r2-lowrank+grouped-lite
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

Low-rank runtime pass relative to L6:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,L6}}}\leq0.75.
$$

Memory no-regression relative to L6:

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

Gradient correctness:

$$
\operatorname{grad\_relerr}<10^{-4}.
$$

### 9.5 可视化

```text
p4_lowrank_package_pareto.svg
p4_lowrank_intermediate_memory_bar.svg
p4_rank_runtime_heatmap.svg
p4_lowrank_step_waterfall.svg
```

---

## 10. P5：piecewise residual-only / mixing decomposition

### 10.1 目的

P5 判断 piecewise local 是否值得继续。v6.14 full package 失败，但 residual-only/noMix 未实现。v6.15 必须拆开 residual 与 mixing。

### 10.2 Candidate packages

```text
PW0-piecewise2-forwardOnly-full-current
PW1-piecewise2-streamingGrad-full-current

PR0-piecewise2-residualOnly-forward
PR1-piecewise2-residualOnly-forward-backward
PR2-piecewise2-streamingGrad-noMix
PR3-piecewise2-residualOnly-custom-knot
PR4-piecewise4-residualOnly-forward-backward

PM0-piecewise2-residual+lowrank-r2
PM1-piecewise2-residual+grouped-vectorized
PM2-piecewise2-residual+grouped-streamed
```

### 10.3 必须记录字段

```text
package
num_bins
active_bins_per_sample
mixing_type
dense_basis_tensor_used
knot_materialized
residual_only
streaming_grad_enabled

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

Residual-only useful:

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

If residual-only passes but full fails, route:

```text
piecewise_mixing_blocker
```

If residual-only fails, route:

```text
piecewise_residual_too_heavy
```

### 10.5 可视化

```text
p5_piecewise_residual_vs_full_pareto.svg
p5_knot_workspace_bar.svg
p5_bin_occupancy_heatmap.svg
p5_dead_bin_fraction_bar.svg
p5_piecewise_mixing_blocker_waterfall.svg
```

---

## 11. P6：reset-v8 package selection

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
  no viable reset-v8
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
p6_reset_v8_scorecard.svg
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
  Best-reset-v8 + ManualAdanLite
  Best-reset-v8 + ManualAdamW
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

P9 只在 P8 official task pass 后打开。它不是 v6.15 主线。

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
R1-ResetV8Solved:
  reset-v8 gets S0/S1 and P7 pass.
  Open task.

R2-ResetV8NearPass:
  reset-v8 gets S2.
  Continue reset engineering, diagnostic task only.

R3-GroupedMemoryRuntimeMerged:
  grouped candidate preserves G2 memory and fixes runtime.
  Continue grouped mainline.

R4-GroupedTradeoffUnresolved:
  grouped candidate remains memory/runtime tradeoff.
  Need custom kernel or stop grouped branch.

R5-LowRankBalancedCandidate:
  low-rank candidate gets S2 or close.
  Continue low-rank mainline.

R6-PiecewiseResidualCandidate:
  piecewise residual-only gets S2 or close.
  Continue piecewise mainline.

R7-RuntimeAttributionIncomplete:
  counters/proxies still unavailable.
  Improve profiler before route claim.

R8-NoViableResetV8:
  no reset-v8 candidate gets S2.
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
p1_runtime_attribution.csv
p2_grouped_tile_stream_sweep.csv
p3_fused_grouped_custom_kernel.csv
p4_lowrank_r2_s2_closure.csv
p5_piecewise_residual_vs_mixing.csv
p6_reset_v8_selection.csv
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
figures/p2_grouped_tile_stream_pareto.svg
figures/p2_materialization_vs_memory.svg
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

### Case A：Grouped memory-preserving runtime fusion 成功

如果某 grouped candidate 同时满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

则 grouped branch 成为 v6.16 主线。如果 P7 通过，可以打开 diagnostic task。

### Case B：Grouped 仍然 memory/runtime tradeoff

如果 loop path memory 好但 runtime 崩，vectorized path runtime 好但 memory 崩，新 candidate 不能合并两者，则 grouped branch 停止作为主线，除非进入真正 custom kernel route。

### Case C：Low-rank r2 达到 S2

如果 low-rank r2 达到 S2，低秩 branch 接替主线。下一步做 one-step probe 与 diagnostic task。

### Case D：Piecewise residual-only 成功

如果 piecewise residual-only 达到 S2 且 full piecewise fails，则下一步修 piecewise + bounded mixing。

### Case E：Piecewise residual-only 也失败

piecewise 分支停止。

### Case F：所有 ResetV8 失败

如果没有 S2 candidate，则停止当前 ResetV6/V7/V8 分支，进入 new primitive family design。

---

## 19. 最终建议

v6.15 的一句话策略是：

$$
\boxed{
\text{不要继续做 memory-only 或 runtime-only 实验，而要把 grouped 的 memory 与 runtime 合并。}
}
$$

同时保留 low-rank r2 作为平衡 fallback，并用 piecewise residual-only 诊断决定 piecewise 是否继续。

如果 v6.15 仍然没有 S0/S1/S2 candidate，就应该停止当前 ResetV7/V8 分支，进入新的 primitive family 设计。
