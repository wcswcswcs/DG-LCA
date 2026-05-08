# DG-KAN v6.12：ResetV4 Runtime Repair、Low-Rank/Chunked Mixing 与 New Bounded-Workspace Primitive Family 详细实验计划

> 本计划基于 v6.11 final real-only run 制定。v6.11 已经输出 `R6-ResetV4TooHeavy`，并保持 `stop_dwm2_patching=1`。DWM2 terminal true single-call/full-step 候选仍未实现，因此 DWM2 不再作为本轮主线。v6.11 的正向信号来自 `B0-lowrank-r4-poly1`：它把 memory ratio mean 从 current DWM2 的 `1.2918` 降到 `1.1285`，memory 改善约 `12.42%`，且 residual effect 真实达标；但它仍未 near-pass，step ratio mean 为 `2.0684`，比 current 更慢。因此 v6.12 的核心目标是：把 reset-v4 的 memory 正信号转化为真正 memory/time survivor，或者给出 bounded-workspace primitive family 仍不可行的路线判定。

---

## 0. 实验整体目标

v6.12 的整体目标不是继续修 DWM2-poly2，也不是打开 task / optimizer / functional correction。v6.12 的目标是回答：

$$
\boxed{
\text{ResetV4 的低秩/分块/局部 residual primitive 能否成为新的轻量 PureKAN 主线？}
}
$$

更具体地说，v6.12 要验证两个问题：

$$
\boxed{
\text{B0-lowrank-r4-poly1 的 memory 改善能否继续压到 near-pass 或 memory-pass？}
}
$$

以及：

$$
\boxed{
\text{B0 的 step time 变慢是否可以通过 fused low-rank mixing、chunked mixing、one-buffer backward 解决？}
}
$$

v6.11 的事实基线如下：

```text
DWM2-current:
  memory ratio mean = 1.2918
  step ratio mean   = 1.9282
  backward ratio mean = 1.5074
  residual/base = 0.0010
  residual effect pass = 0/18

R0-reset-v3-scale002-current:
  memory ratio mean = 1.2981
  step ratio mean   = 1.7915
  backward ratio mean = 1.2523
  residual/base = 0.0200
  residual effect pass = 18/18

B0-lowrank-r4-poly1:
  memory ratio mean = 1.1285
  step ratio mean   = 2.0684
  backward ratio mean = 1.2993
  memory improvement vs current = 12.42%
  step improvement vs current = -6.99%
  residual/base = 0.0200
  residual effect pass = 18/18
  near pass = 0/18
```

v6.12 的最低工程成功标准是：

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

v6.12 不以 $0.80$ 作为唯一成功标准。v6.12 的现实目标是：判断 reset-v4 / bounded-workspace primitive family 是否能接替 DWM2 成为新主线。

---

## 1. 当前问题判断

### 1.1 已经找到了一部分关键原因，但还没有找到最终可修方案

v6.11 相比 v6.10 最大进展是 attribution blocker 被解决了一部分。P1 stable taxonomy attribution 已经通过：

```text
top3 taxonomy gap fraction = 1.0000 / 1.0000 / 1.0000
unknown gap fraction = 0
attribution pass = 18/18
```

这说明现在不再是“完全不知道峰值来自哪里”。但是这还不是最终可修方案，因为 P4 reset component audit 仍然只是来自 full-step phase fields，不是低层 kernel counter；component allocation count / kernel count 仍是 `metric_unavailable`。因此当前状态是：

$$
\boxed{
\text{高层 taxonomy 已经闭合，但低层 kernel/time 修复点还没有完全闭合。}
}
$$

### 1.2 DWM2 小修路线应继续冻结

v6.11 中 DWM2 terminal true single-call/full-step 候选全部 `not_implemented`，route 中 `stop_dwm2_patching=1`。因此 v6.12 不再继续 DWM2 局部 patch，不再新增：

```text
DWM2 coeffgrad-only patch
DWM2 noHidden variant
DWM2 bf16Cache variant
DWM2 bufferReuse wrapper
DWM2 deltaStreaming wrapper
DWM2 torch-level local reduce
```

DWM2 只保留为：

```text
DWM2-current baseline
DWM2 terminal if already implemented elsewhere, diagnostic only
```

v6.12 主线转向 reset-v4 / new bounded-workspace primitive family。

### 1.3 Reset-v4 有真实 memory 信号，但仍太重太慢

`B0-lowrank-r4-poly1` 是 v6.11 最重要的正信号。它让 memory ratio mean 从 `1.2918` 降到 `1.1285`，这是第一次 reset family 出现超过 $10\%$ 的真实 memory improvement。

但它仍然没有 near-pass：

$$
1.1285 > 1.05.
$$

它离 near-pass memory gate 还需要降低：

$$
1-\frac{1.05}{1.1285}\approx6.95\%.
$$

离 memory-pass $1.0$ 还需要降低：

$$
1-\frac{1.0}{1.1285}\approx11.39\%.
$$

离终极 $0.8$ 还需要降低：

$$
1-\frac{0.8}{1.1285}\approx29.11\%.
$$

更严重的是 step time：

$$
r_{\text{step,B0}}=2.0684.
$$

要到 near-pass $1.50$，需要提速：

$$
1-\frac{1.50}{2.0684}\approx27.48\%.
$$

要到 task-open $1.35$，需要提速：

$$
1-\frac{1.35}{2.0684}\approx34.73\%.
$$

因此 v6.12 的重点不是继续证明 residual effect，而是同时解决：

```text
B0 memory 仍差约 7% 才 near-pass
B0 step time 仍差约 27% 才 near-pass
```

### 1.4 Component audit 指出 reset 的主要修复对象

v6.11 的 P4 对 `B0-lowrank-r4-poly1` 做了 component audit，标记为 repair target 的组件是：

```text
residual_transform
mixing_forward
backward_mixing_delta
backward_residual_dx
```

其记录为：

```text
residual_transform:
  component peak MB = 2.1055
  component time ms = 1.5949
  repair target = 1

mixing_forward:
  component peak MB = 2.1055
  component time ms = 1.5949
  repair target = 1

backward_mixing_delta:
  component peak MB = 1.0527
  component time ms = 0.6496
  repair target = 1

backward_residual_dx:
  component peak MB = 1.0527
  component time ms = 0.6496
  repair target = 1
```

这说明 v6.12 的修复不应该继续只调 residual scale，而应聚焦：

```text
low-rank mixing forward
backward mixing delta
residual dx
fused residual + low-rank mixing
chunked low-rank mixing
one-buffer residual backward
```

---

## 2. v6.12 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现项必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许继续 DWM2 局部 patch。DWM2 只作为 baseline 或 terminal diagnostic，不作为主线。

第三，不允许把 memory improvement 写成 success，除非同时满足 step time 和 gradient correctness gate。

第四，不允许只调 residual scale。v6.11 已经证明 residual effect 可以达标。v6.12 必须改变 workspace / runtime model。

第五，不允许把 ManualLinear reference 当 PureKAN success。它只能作为 lower-bound baseline。

第六，P0-P5 没有 S0/S1/S2 survivor 时，不允许打开：

```text
task re-entry
optimizer exploration
LightSmooth
functional correction
3-seed / 5-seed / 10-seed confirm
```

第七，不允许把 P4 component audit 当作低层 kernel counter。v6.12 必须补充真实 kernel/allocation counter，否则只能作 phase-level diagnostic。

---

## 3. 核心假设

### H1：B0 memory 改善来自 low-rank mixing，但 step time 变慢来自 unfused two-stage low-rank path

`B0-lowrank-r4-poly1` 的 memory 改善约 $12.42\%$，说明 low-rank mixing 对 memory 有价值。但 step time 比 current 更慢约 $6.99\%$。H1 假设：B0 的慢来自低秩路径被拆成多个小 matmul / elementwise / update kernels，而不是 low-rank idea 本身无效。

H1 成立需要观察到：

```text
lowrank_factor1_time + lowrank_factor2_time 占 step time >= 25%
lowrank intermediate materialization 明显
kernel_count_mixing 增加
allocation_count_mixing 增加
```

H1 的修复标准是 fused low-rank path 相对 B0 满足：

$$
\frac{T_{\text{step,fused-lowrank}}}{T_{\text{step,B0}}}\leq0.80,
$$

同时 memory 不恶化超过 $5\%$：

$$
\frac{M_{\text{fused-lowrank}}}{M_{\text{B0}}}\leq1.05.
$$

如果 H1 成立且修复有效，B0 可以继续作为主线。

---

### H2：B0 离 memory near-pass 只差约 7%，可通过 chunked low-rank mixing 或 one-buffer residual backward 补足

B0 的 memory ratio mean 为 $1.1285$。Near-pass 要求 $1.05$。H2 假设：低秩方向已经接近，剩余 gap 可能来自 mixing workspace 或 backward residual buffer，可以通过 chunked low-rank mixing 或 one-buffer residual backward 降低。

H2 成立要求：

$$
r_{\text{mem,new}}\leq1.05,
$$

且：

$$
r_{\text{step,new}}\leq1.50.
$$

如果 memory 降到 near-pass 但 step time 仍 $>1.50$，不能开 task，只能继续 runtime repair。

---

### H3：Reset-v4 的主要 hidden blocker 是 mixing/workspace lifetime，而不是 residual expressivity

v6.11 所有 measured reset-v4 residual effect 都达标，因此 residual expressivity 已经不是核心 blocker。H3 假设：当前 reset 的真正问题是 mixing/workspace lifetime：

```text
mixing output live too long
low-rank intermediate live too long
chunked mix not actually reducing live set
backward mixing delta materializes full tensor
```

H3 成立标准是 P1/P2 attribution 中 mixing/workspace 相关 source 至少解释：

$$
\frac{M_{\text{mixing/workspace}}}{G_{\text{peak}}}\geq0.30.
$$

如果 H3 成立，P3/P4 必须优先修 mixing，而不是 residual transform。

---

### H4：Chunked mixing 可以降低 memory，但可能增加 time；需要找到 chunk size Pareto

Chunked mixing 理论上可以降低 output/mixing workspace，但可能增加 kernel launch 和 loop overhead。H4 假设存在一个 chunk size $c^\*$，使得 memory 与 step time 同时接近 near-pass。

候选 chunk size：

```text
c = 4, 8, 16, 32, 64
```

H4 成立标准是某个 chunk size 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果 chunk 越小 memory 越低但 time 越高，必须报告 memory-time Pareto，而不能只取最低 memory。

---

### H5：Grouped / block-diagonal mixing 可能比 low-rank mixing 更适合 bounded workspace

Low-rank mixing 降低 memory 但 step 慢。H5 假设：grouped/block-diagonal mixing 可以减少 workspace 与 computation，同时保持 residual effect。

候选：

```text
group_count = 4, 8, 16
block_size matched to hidden_dim
within-group residual + mixing
optional cross-group lightweight correction
```

H5 成立标准：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

且 residual effect pass：

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02.
$$

如果 grouped mixing memory/time 过但 residual ablation 影响为 0，则不算 PureKAN success。

---

### H6：如果 reset-v5 仍不能 near-pass，应启动 new primitive family design，而不是继续 reset patch

H6 是 Stop/Go 假设。如果本轮完成：

```text
low-rank fused repair
chunked mixing sweep
grouped/block mixing
piecewise local residual
component audit
```

仍没有 S0/S1/S2，则 v6.12 应输出：

```text
R7-NoViableBoundedReset
```

下一轮必须进入 new primitive family，不再围绕 poly1 low-rank reset 做 patch。

---

## 4. 实验阶段总览

v6.12 分为十二个阶段：

```text
P0: v6.11 reproduction and reset-lineage contract
P1: low-rank reset attribution and kernel audit
P2: fused low-rank reset runtime repair
P3: chunked low-rank mixing sweep
P4: grouped / block-diagonal bounded reset
P5: piecewise local bounded reset
P6: reset-v5 package selection
P7: one-step loss and residual probe
P8: limited task re-entry gate
P9: optimizer exploration remains gated
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P6 是核心。P7 只对 S0/S1/S2 candidate 运行。P8-P10 默认关闭。

---

## 5. P0：v6.11 reproduction and reset-lineage contract

### 5.1 目的

确认 v6.12 与 v6.11 baseline 可比，并确认 DWM2 patch 已冻结、reset lineage 真实。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-current-baseline
B0-lowrank-r4-poly1
B1-lowrank-r8-poly1
A0-inplace-poly1-chunked-mix-c16
A1-inplace-poly1-chunked-mix-c32
ResetV5-fused-lowrank-r4, if implemented
ResetV5-chunked-lowrank, if implemented
ResetV5-grouped-mixing, if implemented
ResetV5-piecewise-local, if implemented
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
v611_memory_ratio_mean
v612_memory_ratio_mean
v611_step_ratio_mean
v612_step_ratio_mean
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
is_dwm2_patch
dwm2_patch_allowed
```

### 5.4 通过标准

Reproduction pass:

$$
|r_{\text{mem,v612}}-r_{\text{mem,v611}}|\leq0.05.
$$

$$
|r_{\text{step,v612}}-r_{\text{step,v611}}|\leq0.15.
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

## 6. P1：low-rank reset attribution and kernel audit

### 6.1 目的

P1 解决 v6.11 P4 不足：P4 只用 full-step phase fields 做 component audit，缺少真实 allocation/kernel counter。P1 要把 B0 的 step 慢与 memory 剩余 gap 拆到具体 component。

### 6.2 测量对象

```text
MLP-manual-linear-reference
DWM2-current-baseline
B0-lowrank-r4-poly1
B1-lowrank-r8-poly1
A0-inplace-poly1-chunked-mix-c16
A1-inplace-poly1-chunked-mix-c32
```

### 6.3 Component phases

```text
phase_residual_transform
phase_lowrank_factor_V
phase_lowrank_factor_U
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

Component 是 repair target 如果满足至少一项：

$$
\frac{T_{\text{component}}}{T_{\text{step}}}\geq0.25.
$$

或：

$$
\frac{M_{\text{component}}}{G_{\text{peak}}}\geq0.25.
$$

其中：

$$
G_{\text{peak}}=M_{\text{peak,candidate}}-M_{\text{peak,MLP}}.
$$

P1 通过要求至少识别一个 primary repair target：

```text
primary_repair_target != unknown
```

并且该 target 的 time 或 memory fraction 达到上述阈值。

### 6.6 可视化

```text
p1_component_runtime_waterfall.svg
p1_component_memory_waterfall.svg
p1_component_fraction_heatmap.svg
p1_lowrank_factor_runtime_bar.svg
p1_mixing_workspace_attribution.svg
```

---

## 7. P2：fused low-rank reset runtime repair

### 7.1 目的

P2 直接修复 B0 的 step time。B0 memory 有正信号，但 step 太慢。P2 目标是把 B0 的 step ratio 从 `2.0684` 拉到 `<=1.50`，同时不丢掉 memory improvement。

### 7.2 Candidate packages

```text
L0-B0-lowrank-r4-current

L1-fused-lowrank-forward:
  fuse residual transform + V projection + U projection where possible

L2-fused-lowrank-backward:
  fuse backward mixing delta + residual dx

L3-fused-lowrank-forward-backward:
  combine L1 and L2

L4-lowrank-no-intermediate:
  avoid materializing intermediate low-rank activation if possible

L5-lowrank-onebuffer:
  reuse a single low-rank intermediate buffer

L6-lowrank-r2-fast:
  rank=2 diagnostic for lower runtime

L7-lowrank-r4-compiled:
  torch.compile or Triton fused path if stable

L8-lowrank-r4-custom-kernel:
  custom fused kernel if implemented
```

### 7.3 必须记录字段

```text
package
rank
implementation_status
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
memory_improvement_vs_B0
step_improvement_vs_B0
memory_improvement_vs_DWM2_current
step_improvement_vs_DWM2_current
intermediate_lowrank_MB
lowrank_kernel_count
lowrank_allocation_count
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 7.4 判断标准

Fused low-rank useful:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,B0}}}\leq0.80.
$$

Memory must not regress:

$$
\frac{M_{\text{new}}}{M_{\text{B0}}}\leq1.05.
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
```

---

## 8. P3：chunked low-rank mixing sweep

### 8.1 目的

P3 验证 chunked mixing 是否可以把 memory ratio 从 `1.1285` 压到 near-pass，同时控制 step time。

### 8.2 Candidate packages

```text
C0-B0-lowrank-r4-current
C1-lowrank-r4-chunk4
C2-lowrank-r4-chunk8
C3-lowrank-r4-chunk16
C4-lowrank-r4-chunk32
C5-lowrank-r4-chunk64
C6-lowrank-r8-chunk16
C7-lowrank-r2-chunk16
```

### 8.3 必须记录字段

```text
package
rank
chunk_size
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
mixing_workspace_MB
chunk_boundary_overhead_ms
chunk_kernel_count
chunk_allocation_count
memory_improvement_vs_B0
step_improvement_vs_B0
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
```

### 8.4 判断标准

Chunked low-rank useful:

$$
r_{\text{mem}}\leq1.05.
$$

且：

$$
r_{\text{step}}\leq1.50.
$$

如果某 candidate memory 最好但 step time 过差，则只能作为 diagnostic。最终选择按 Pareto，而不是单一最低 memory。

### 8.5 可视化

```text
p3_chunk_size_pareto.svg
p3_chunk_size_memory_curve.svg
p3_chunk_size_step_curve.svg
p3_rank_chunk_heatmap.svg
```

---

## 9. P4：grouped / block-diagonal bounded reset

### 9.1 目的

P4 测试 grouped/block-diagonal mixing 是否比 low-rank 更适合 bounded workspace。

### 9.2 Candidate packages

```text
G0-grouped-g4-poly1
G1-grouped-g8-poly1
G2-grouped-g16-poly1
G3-grouped-g4-piecewise2
G4-grouped-g8-piecewise2
G5-blockdiag-b4-poly1
G6-blockdiag-b8-poly1
G7-grouped-g8-crossgroup-light
```

### 9.3 必须记录字段

```text
package
group_count
block_size
cross_group_enabled
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
group_workspace_MB
cross_group_workspace_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_ablation_delta_loss
residual_ablation_delta_logit
residual_effect_pass
```

### 9.4 判断标准

Grouped reset near-pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Residual effect:

$$
\frac{\|s r(x)\|}{\|x\|}\geq0.02.
$$

And either:

$$
|\Delta L_{\text{residual ablation}}|>10^{-4},
$$

or:

$$
|\Delta \text{logit}_{\text{residual ablation}}|>10^{-4}.
$$

### 9.5 可视化

```text
p4_grouped_reset_pareto.svg
p4_group_count_heatmap.svg
p4_residual_effect_by_group.svg
p4_workspace_by_group.svg
```

---

## 10. P5：piecewise local bounded reset

### 10.1 目的

P5 验证 local piecewise residual 是否能避免 low-rank/chunked mixing 的 runtime 开销，同时保持 residual 非平凡。

### 10.2 Candidate packages

```text
P0-piecewise2-local-forwardOnly-diagnostic
P1-piecewise2-streamingGrad
P2-piecewise4-streamingGrad
P3-piecewise2-chunkedMix
P4-piecewise2-groupedMix
P5-piecewise2-lowrankMix-r4
```

### 10.3 必须记录字段

```text
package
num_bins
active_bins_per_sample
streaming_grad_enabled
mixing_type
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
knot_workspace_MB
knot_grad_workspace_MB
grad_relerr_max
grad_cos_min
residual_over_base
residual_effect_pass
bin_occupancy_entropy
dead_bin_fraction
out_of_grid_fraction
```

### 10.4 判断标准

Piecewise candidate pass:

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

### 10.5 可视化

```text
p5_piecewise_pareto.svg
p5_bin_occupancy_heatmap.svg
p5_dead_bin_fraction_bar.svg
p5_out_of_grid_curve.svg
```

---

## 11. P6：reset-v5 package selection

### 11.1 目的

P6 汇总 P2-P5，选择是否存在 reset-v5 survivor。

### 11.2 Survivor 类型

```text
S0:
  r_mem < 1.0 and r_step <= 1.20

S1:
  r_mem < 1.0 and r_step <= 1.35

S2:
  r_mem <= 1.05 and r_step <= 1.50 and residual effect pass

S3:
  memory improves but step fails

S4:
  step improves but memory fails

S5:
  residual effect fails

S6:
  gradient correctness fails

S7:
  no viable reset
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
p6_reset_v5_scorecard.svg
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
  Best-reset-v5 + ManualAdanLite
  Best-reset-v5 + ManualAdamW
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

P9 只在 P8 official task pass 后打开。它不是 v6.12 主线。

如果打开，只比较：

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
R1-ResetV5Solved:
  reset-v5 gets S0/S1 and P7 pass.
  Open task.

R2-ResetV5NearPass:
  reset-v5 gets S2.
  Continue reset engineering, diagnostic task only.

R3-LowRankRuntimeRepairNeeded:
  low-rank memory is good but step time still fails.
  Continue low-rank runtime fusion.

R4-MixingWorkspaceBlocker:
  mixing/workspace dominates peak.
  Continue chunk/group/lowrank mixing redesign.

R5-ResidualTransformBlocker:
  residual transform dominates.
  Need new residual primitive.

R6-PiecewiseLocalCandidate:
  piecewise local reset gets S0/S1/S2.
  Switch mainline to piecewise local.

R7-NoViableBoundedReset:
  no reset-v5 candidate gets S2.
  Need new primitive family beyond current reset set.

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
F6_chunked_time_fail
F7_grouped_memory_fail
F8_piecewise_stability_fail
F9_task_gated
F10_optimizer_gated
F11_functional_gated
F12_fake_or_proxy_violation
F13_artifact_missing
```

### artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p1_reset_component_audit.csv
p2_fused_lowrank_repair.csv
p3_chunked_lowrank_sweep.csv
p4_grouped_bounded_reset.csv
p5_piecewise_local_reset.csv
p6_reset_v5_selection.csv
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
figures/p1_component_memory_waterfall.svg
figures/p2_lowrank_package_pareto.svg
figures/p3_chunk_size_pareto.svg
figures/p4_grouped_reset_pareto.svg
figures/p5_piecewise_pareto.svg
figures/p6_family_comparison_pareto.svg
figures/p7_loss_before_after.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 18. 成功与失败解释规则

### Case A：B0 修复后 near-pass

如果 fused low-rank / chunked low-rank 让 B0 达到 S2 或更好，reset-v5 继续作为主线。

### Case B：B0 memory 好但 step time 仍失败

如果 memory $\leq1.05$，但 step time $>1.50$，说明 low-rank memory model 有用但 runtime 不行。下一步继续 low-rank fusion，不开 task。

### Case C：Grouped / block mixing 成功

如果 grouped/block-diagonal reset 达到 S0/S1/S2，则切换主线到 grouped reset。

### Case D：Piecewise local 成功

如果 piecewise local reset 达到 S0/S1/S2 且稳定性过 gate，则切换到 piecewise local。

### Case E：所有 reset-v5 失败

如果没有 S2 candidate，则停止当前 reset-v4/v5 分支，进入 new primitive family design。

---

## 19. 最终建议

v6.12 的一句话策略是：

$$
\boxed{
\text{以 B0-lowrank-r4-poly1 的 memory 改善为线索，修 runtime；同时并行验证 grouped 与 piecewise local bounded primitive。}
}
$$

不要再继续 DWM2 小修。DWM2 已经冻结，除非真正 terminal full-step kernel 已经实现。v6.12 的主线是 bounded-workspace primitive family reset。

如果 v6.12 仍然没有 S0/S1/S2，那么应该正式进入新 primitive family 设计，而不是继续在 low-rank/poly1 reset 上 patch。
