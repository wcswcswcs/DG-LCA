# DG-KAN v6.19：GT4 S2 恢复后的真实 Kernel Margin Repair、Protocol Unification 与 Diagnostic Task Reopen 详细实验计划

> 本计划基于 v6.18 final real-only run 制定。v6.18 的最终 route 是 `R6-S2NotRestored`。本轮不是 fused grouped 路线失败，而是 **S2 margin restoration / kernel margin repair 失败**：v6.18 真实复现了 v6.16 GT4 baseline 的 memory 与 step 可比性，证明 baseline step drift 已解决；但在 P2/P3 的 repair protocol 下，GT4 baseline step 仍略高于 S2 gate，E1 hidden-cache recompute 虽然降低 memory，却进一步恶化 step/backward，因此没有 S0/S1/S2 candidate，P5-P10 正确 gate。v6.19 的目标是停止 recompute 型 memory repair，转向真实 kernel margin repair：grad/update buffer reuse、launch/update/forward fastpath、buffer lifetime trim，并在恢复 S2 后重新打开 diagnostic task gap attribution。

---

## 0. 实验整体目标

v6.19 的整体目标不是回到 DWM2，不是重新搜索 low-rank / piecewise，也不是现在就大规模优化 optimizer。v6.19 的整体目标是围绕 fused grouped GT4 完成一次真实 margin 修复：

$$
\boxed{
\text{恢复并稳定 GT4 的 S2，然后以真实 kernel margin repair 推向 S1/S0。}
}
$$

具体来说，v6.19 要回答三个问题。

第一个问题：

$$
\boxed{
\text{为什么 P0 中 GT4 能复现 v6.16 S2，而 P2/P3 protocol 下 baseline step 又略高于 S2 gate？}
}
$$

第二个问题：

$$
\boxed{
\text{能否通过 non-recompute 的 kernel margin repair，把 step 压回 } \leq 1.50 \text{ 并进一步压到 } \leq1.35？
}
$$

第三个问题：

$$
\boxed{
\text{S2 恢复后，GT4 diagnostic task gap 到底来自表达力、优化器、泛化，还是训练预算？}
}
$$

v6.18 的真实事实基线如下：

```text
P0 A2-GT4-v616-runner-reference:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4774
  backward ratio mean = 0.7571
  grad relerr max     = 1.92e-08
  v6.16 reproduction pass = 1

P0 A3-E0-GT4-v617-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4845
  backward ratio mean = 0.7616
  grad relerr max     = 1.92e-08
  v6.16 reproduction pass = 1

P0 A6-GT4-v617-kernel-with-v616-timing-protocol:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4727
  backward ratio mean = 0.7548
  grad relerr max     = 1.92e-08
  v6.16 reproduction pass = 1
```

这说明 baseline step drift 在 P0 side-by-side 层面已经解决，GT4 fused grouped kernel 仍然可以达到 S2 附近的真实运行状态。但 P2/P3 的 repair protocol 中又出现了 step gate 失败：

```text
P2 M0-GT4-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5228
  backward ratio mean = 0.7849
  forward ratio mean  = 1.2331
  S2 = 0

P3 T0-GT4-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5375
  backward ratio mean = 0.8365
  forward ratio mean  = 1.1980
  S2 = 0
```

E1 hidden-cache recompute 的情况是：

```text
P2 M8-E1-recompute-hidden-cache:
  memory ratio mean   = 1.0234
  step ratio mean     = 1.5907
  backward ratio mean = 0.9419
  forward ratio mean  = 1.1889
  S2 = 0
```

因此 v6.19 的最低目标是稳定恢复 S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

并保持：

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

v6.19 的主要目标是拿到 S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

v6.19 的强目标是：

$$
r_{\text{mem}}\leq0.95,
$$

$$
r_{\text{step}}\leq1.25.
$$

终极 memory 目标仍然是：

$$
r_{\text{mem}}\leq0.80.
$$

但 v6.19 不把 $0.80$ 当作本轮唯一 gate。本轮的核心是：先把 S2 作为稳定工程底座恢复，再尝试 S1/S0，并且恢复 diagnostic task 以解释 task gap。

---

## 1. 当前实验进展与问题判断

### 1.1 已经确定的正向进展

v6.18 不是完全失败。它确认了三件重要事实。

第一，v6.16 GT4 baseline 可以在 v6.18 的 P0 side-by-side protocol 中复现，`A2/A3/A5/A6` 的 step ratio 都在 `1.47-1.48` 附近，满足 S2 的 `step <= 1.50`。这说明 fused grouped kernel 本体没有坏，v6.16 的 S2 不是偶然或 fake/proxy。

第二，v6.18 显式记录了 W&B absolute memory 字段，包括：

```text
memory/KAN_peak_MB
memory/MLP_peak_MB
memory/peak_allocated_MB
memory/peak_reserved_MB
memory/ratio_vs_MLP
```

这些值来自本地 profiler CSV，不是只记录 ratio。因此之后分析 memory margin 不再缺 absolute memory 证据。

第三，E1 hidden-cache recompute 的负结果非常清楚。它证明 recompute 可以降低 memory，但代价是 backward/step 变慢。这不是梯度错误，而是实现策略本身不适合当前 margin repair。

### 1.2 v6.18 的真实失败

v6.18 的 route 是：

```text
R6-S2NotRestored
```

最佳 candidate 是：

```text
C3-safe-memory+step-combo = M8-E1-recompute-hidden-cache
```

它的结果是：

```text
memory ratio = 1.0234
step ratio   = 1.5907
backward ratio = 0.9419
survivor type = S3
```

也就是说，E1 是 memory-better / step-fail candidate。它不是 S2，更不是 S1/S0。P5 one-step、P6 diagnostic task、P7 optimizer diagnostic、P8 official task、P9 optimizer exploration、P10 functional correction 全部关闭是正确的。

### 1.3 当前问题所在

当前真正的问题不是：

```text
Triton fused grouped kernel 是否可行
manual backward 是否正确
W&B 是否没记显存
fake/proxy 是否污染
DWM2 是否还该继续
low-rank/piecewise 是否是主线
```

这些都已经有明确结论。当前真正问题是：

$$
\boxed{
\text{GT4 的 S2 margin 太薄，而 v6.18 的 repair 方向选错了。}
}
$$

更细分地说，有三个 blocker：

```text
B1: P0 protocol 和 P2/P3 repair protocol 下的 step ratio 有 3%-4% 差异。
B2: E1 recompute 降低 memory 但伤害 backward/step。
B3: 真实可能有效的 repair，如 grad/update buffer reuse、launch-fused step、forward/update fastpath，仍未实现。
```

因此 v6.19 的任务不是继续 E1，而是实现 v6.18 中未完成的真实 kernel margin repair。

### 1.4 是否在正确道路上

项目仍在正确道路上，因为 v6.16 已经给出了真实 S2 fused grouped survivor，v6.18 又确认该 baseline 可以在 side-by-side protocol 中复现。当前失败不是路线失败，而是 margin repair 没实现到位。

但如果 v6.19 继续做 hidden recompute、memory-only selection 或普通 task/optimizer 大扫，就会偏离正确道路。正确道路应该是：

```text
1. 统一 P0/P2/P3 timing protocol；
2. 恢复稳定 S2；
3. 实现 non-recompute kernel margin repair；
4. S2 后重新打开 diagnostic task；
5. S1/S0 后再打开 official task 和系统 optimizer exploration。
```

---

## 2. 离目标还差多远

### 2.1 以 P0 复现的 GT4 计算

P0 A6 的结果是：

$$
r_{\text{mem}}=1.0312,
$$

$$
r_{\text{step}}=1.4727.
$$

它已经满足 S2：

$$
r_{\text{mem}}\leq1.05,\quad r_{\text{step}}\leq1.50.
$$

到 S1/S0 memory gate：

$$
r_{\text{mem}}<1.00,
$$

需要降低：

$$
1-\frac{1.00}{1.0312}\approx3.03\%.
$$

到 S1 step gate：

$$
r_{\text{step}}\leq1.35,
$$

需要提速：

$$
1-\frac{1.35}{1.4727}\approx8.33\%.
$$

这是一个小到中等幅度的工程 margin，而不是之前那种 20%-80% 的路线级差距。

### 2.2 以 P2 M0 baseline 计算

P2 M0 的结果是：

$$
r_{\text{mem}}=1.0312,
$$

$$
r_{\text{step}}=1.5228.
$$

到 S2 step gate：

$$
1-\frac{1.50}{1.5228}\approx1.50\%.
$$

到 S1 step gate：

$$
1-\frac{1.35}{1.5228}\approx11.35\%.
$$

因此如果以 P2 protocol 为准，S2 只差约 $1.5\%$ step speedup；S1 需要约 $11.35\%$ step speedup 与约 $3.03\%$ memory reduction。

### 2.3 以 E1 best-memory candidate 计算

E1 的结果是：

$$
r_{\text{mem}}=1.0234,
$$

$$
r_{\text{step}}=1.5907.
$$

到 S2 step gate：

$$
1-\frac{1.50}{1.5907}\approx5.70\%.
$$

到 S1 step gate：

$$
1-\frac{1.35}{1.5907}\approx15.13\%.
$$

E1 memory 更接近 S1 memory gate，但 step 远离 S2。因此 E1 不应作为主线。

### 2.4 task / optimizer 距离

v6.18 没有 S2 candidate，所以没有 diagnostic task。v6.16 的 diagnostic task gap 仍是参考，但不能用 v6.18 更新：

```text
GT4+ManualAdanLite val acc = 0.6302
MLP-autograd val acc       = 0.6981
gap ≈ -6.79 percentage points
```

这说明即使 S1/S0 达成，task quality 仍需要分析。但 optimizer 不能在 S2 丢失时大规模展开。

---

## 3. v6.19 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。v6.19 不允许新增 DWM2 patch。

第三，low-rank、piecewise、Python grouped loop 不作为主线。它们只能作为历史 reference，不进入 route selection。

第四，E1 hidden-cache recompute 不允许进入主 combo，除非它同时满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

第五，不允许按 best memory 选择 candidate。v6.18 的教训是 best memory candidate 可能是 S3。v6.19 必须按 survivor type 和 Pareto 选择。

第六，没有 S2 时，不允许打开 one-step、diagnostic task、official task、optimizer exploration、functional correction。

第七，S2 可以打开 diagnostic task，但不能叫 official task success。Official task 必须等 S1/S0。

第八，任何 optimizer / regularization diagnostic 必须在 S2 恢复之后运行；如果没有 S2，optimizer 结果会被 efficiency gate 混淆。

第九，任何 cross-group / expressivity repair 必须保持：

```text
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
```

第十，所有 repair 必须记录 absolute memory MB，不允许只报告 ratio。

---

## 4. 核心假设

### H1：v6.18 未恢复 S2 的主要原因是 repair protocol step overhead，而不是 GT4 kernel 失效

P0 已经显示 GT4 在多个 runner/kernel/timing 组合下可以复现约 `1.47-1.48` 的 step ratio。但 P2/P3 protocol 的 baseline 是 `1.5228-1.5375`。H1 假设：P2/P3 中额外 wrapper、measurement phase、component logging、repair package dispatch 或 timing sync 带来了额外 step overhead。

H1 成立要求 P1 识别到：

```text
P2/P3 protocol extra overhead >= 1.5% step ratio
```

并能归因到至少一个来源：

```text
extra cuda sync
extra torch op dispatch
component wrapper overhead
allocation proxy logging overhead
repair package branch overhead
timing protocol mismatch
```

H1 的修复目标是：在 P2/P3 style protocol 下，GT4 baseline 也恢复到：

$$
r_{\text{step}}\leq1.50.
$$

### H2：hidden-cache recompute 不是合适的 margin repair

H2 已被 v6.17/v6.18 强烈支持。E1 降低 memory，但显著增加 backward 和 step。v6.19 只把 E1 作为 diagnostic，不作为 combo 默认组件。

H2 的量化判定：

$$
M_{\text{E1}}<M_{\text{GT4}},
$$

但：

$$
T_{\text{step,E1}}>T_{\text{step,GT4}},
$$

且：

$$
T_{\text{backward,E1}}>T_{\text{backward,GT4}}.
$$

如果 v6.19 仍复现该模式，则 E1 分支停止，不再组合。

### H3：GT4 memory margin 应通过 buffer lifetime 和 update/grad buffer reuse 修复

GT4 memory gap 只有约 $3\%$。H3 假设：相比 hidden recompute，更适合的 memory repair 是：

```text
grad_mix buffer reuse
grad_poly buffer reuse
update_temp reuse
short-lived grouped output free
triton scratch lifetime trim
reserved/allocated gap trimming
```

H3 成立要求至少一个 repair 满足：

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985,
$$

并且：

$$
\frac{T_{\text{step,new}}}{T_{\text{GT4}}}\leq1.02.
$$

也就是 memory 至少降 $1.5\%$，step 不恶化超过 $2\%$。

### H4：GT4 step margin 应通过 forward/update/launch fastpath 修复，而不是继续优化 backward

GT4 的 backward ratio 已低于 MLP，当前 step gap 更可能来自：

```text
forward path
update prep
optimizer update
kernel launch overhead
torch wrapper overhead
layout conversion
timing/sync overhead
```

H4 成立要求 component timing 显示：

$$
\frac{T_{\text{forward+update+launch}}}{T_{\text{step overhead}}}\geq0.50.
$$

H4 修复成立要求至少一个 step repair 满足：

$$
\frac{T_{\text{step,new}}}{T_{\text{GT4}}}\leq0.96,
$$

且：

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq1.02.
$$

### H5：S2 恢复后才能继续 task gap attribution

v6.16 的 diagnostic task gap 仍然重要，但 v6.18 没有 S2，因此不能更新 task conclusion。H5 要求：

```text
If S2 restored:
  open one-step
  if one-step pass, open diagnostic task
Else:
  no task, no optimizer
```

Diagnostic task 必须记录：

```text
feature rank
margin
classwise accuracy
ECE
NLL
train-val gap
update statistics
role-wise update share
```

不能只看 val/test accuracy。

### H6：如果 v6.19 仍不能恢复稳定 S2，则需要先做 fused grouped measurement protocol repair，而不是 expressivity/optimizer

如果 v6.19 完成：

```text
protocol overhead attribution
non-recompute memory repair
step fastpath repair
safe combo selection
```

仍没有 S2，则不能开 optimizer。下一轮应回到 kernel/protocol 本身，或者考虑接受 v6.16 S2 protocol 为 canonical measurement，并重新定义 gate。

---

## 5. 实验阶段总览

v6.19 分为十二个阶段：

```text
P0: v6.18 reproduction and canonical timing protocol lock
P1: protocol overhead and margin attribution
P2: non-recompute step-preserving memory repair
P3: memory-preserving step repair
P4: safe combo selection by survivor type
P5: one-step probe if S2/S1/S0
P6: diagnostic task gap attribution if S2 restored
P7: small optimizer / regularization diagnostic under S2
P8: official task re-entry if S1/S0
P9: optimizer exploration remains gated unless official task passes
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P4 是硬核心。P5-P7 只有 S2 以上才运行。P8-P10 默认关闭。

---

## 6. P0：v6.18 reproduction and canonical timing protocol lock

### 6.1 目的

P0 要锁定 canonical timing protocol，避免 P0、P2、P3 出现不可解释的 step ratio 差异。v6.19 不能在 measurement protocol 不统一的情况下判断 repair 成败。

### 6.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-runner-reference
E0-GT4-v617-baseline
GT4-v616-kernel-with-v617-runner
GT4-v617-kernel-with-v616-timing-protocol
P2-M0-GT4-baseline-reproduced
P3-T0-GT4-baseline-reproduced
E1-recompute-hidden-cache-diagnostic
```

### 6.3 必须记录字段

```text
variant
runner_version
kernel_version
timing_protocol
repair_protocol_enabled
component_logging_enabled
dataset
batch_size
depth
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
uses_triton_kernel
manual_backward_available
nonKAN_param_count
grad_relerr_max
grad_cos_min
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
peak_allocated_MB
peak_reserved_MB
torch_op_count
cuda_sync_count
component_call_count
warmup_steps
measure_steps
canonical_protocol_pass
```

### 6.4 判断标准

Canonical protocol pass requires:

$$
|r_{\text{step,P0-GT4}}-r_{\text{step,P2-M0}}|\leq0.03.
$$

and:

$$
|r_{\text{step,P0-GT4}}-r_{\text{step,P3-T0}}|\leq0.03.
$$

GT4 S2 reproduction pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

If canonical protocol fails, P1 must classify the source before P2/P3 results are treated as definitive.

### 6.5 可视化

```text
p0_protocol_step_ratio_matrix.svg
p0_protocol_memory_ratio_matrix.svg
p0_step_drift_waterfall.svg
p0_canonical_protocol_pass_heatmap.svg
```

---

## 7. P1：protocol overhead and margin attribution

### 7.1 目的

P1 要解释 GT4 的 remaining step and memory margin。v6.18 已经解决 baseline drift，但 P2/P3 protocol 下 S2 仍不稳定，因此 P1 必须拆出额外 overhead。

### 7.2 测量对象

```text
GT4-canonical-baseline
P2-M0-GT4-baseline
P3-T0-GT4-baseline
E1-recompute-hidden-cache
MLP-manual-linear-reference
```

### 7.3 Component phases

```text
phase_forward_fused_grouped
phase_forward_residual_scale
phase_forward_grouped_mix
phase_loss_delta
phase_backward_fused_grouped
phase_backward_recompute_layer_input
phase_backward_dx
phase_backward_grad_mix
phase_backward_grad_poly
phase_update_prep
phase_optimizer_update
phase_timing_sync
phase_component_logging
phase_cleanup
```

### 7.4 必须记录字段

```text
variant
phase_name
phase_time_ms
phase_peak_MB
phase_alloc_count
phase_torch_op_count
phase_cuda_sync_count
phase_component_call_count
phase_kernel_launch_proxy_count
phase_memory_source_top1
phase_memory_source_top2
phase_memory_source_top3
unknown_time_fraction
unknown_memory_fraction
```

E1-specific fields:

```text
hidden_cache_saved_MB
recompute_time_ms
backward_extra_time_ms_vs_GT4
memory_saved_per_extra_ms
recompute_kernel_count
recompute_torch_op_count
```

### 7.5 判断标准

P1 protocol overhead pass:

```text
primary_protocol_overhead_source identified
unknown_time_fraction <= 0.10
```

E1 continuation pass:

$$
\frac{M_{\text{E1}}}{M_{\text{GT4}}}\leq0.985,
$$

and:

$$
\frac{T_{\text{step,E1}}}{T_{\text{step,GT4}}}\leq1.02.
$$

If E1 fails, it is excluded from P4 combo.

### 7.6 可视化

```text
p1_protocol_overhead_waterfall.svg
p1_gt4_step_margin_waterfall.svg
p1_gt4_memory_margin_waterfall.svg
p1_e1_memory_saved_per_extra_ms.svg
p1_phase_time_stacked_bar.svg
```

---

## 8. P2：non-recompute step-preserving memory repair

### 8.1 目的

P2 要实现 v6.18 未实现的真实 memory repair，而且不允许通过 expensive recompute 换 memory。

### 8.2 Candidate packages

```text
M0-GT4-canonical-baseline

M1-grad-mix-buffer-reuse:
  reuse grad_mix buffer after accumulation

M2-grad-poly-buffer-reuse:
  reuse grad_poly buffer and avoid duplicate grad temp

M3-update-temp-reuse:
  reuse update temp and avoid separate update materialization

M4-triton-scratch-lifetime-trim:
  release or reuse triton scratch earlier

M5-reserved-memory-trim:
  reduce reserved-unallocated gap and allocator padding

M6-short-lived-output-free:
  shorten grouped output lifetime after backward consumes it

M7-grad-update-buffer-combo:
  M1 + M2 + M3

M8-safe-memory-combo:
  M1 + M2 + M3 + M4 + M6

M9-E1-recompute-hidden-cache:
  diagnostic only, excluded from default combo
```

### 8.3 必须记录字段

```text
package
implementation_status
uses_recompute
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
memory_improvement_vs_GT4
step_change_vs_GT4
peak_allocated_MB
peak_reserved_MB
reserved_minus_allocated_MB
grad_mix_MB
grad_poly_MB
update_temp_MB
triton_scratch_MB
grouped_output_lifetime_ms
kernel_count_total
torch_op_count
allocation_proxy_count
grad_relerr_max
grad_cos_min
```

### 8.4 判断标准

Memory useful:

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985.
$$

Step-preserving:

$$
\frac{T_{\text{step,new}}}{T_{\text{GT4}}}\leq1.02.
$$

S2:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

A memory repair that worsens step by more than $2\%$ is S3 diagnostic only.

### 8.5 可视化

```text
p2_memory_repair_pareto.svg
p2_memory_source_reduction_bar.svg
p2_step_preservation_bar.svg
p2_recompute_vs_nonrecompute_tradeoff.svg
```

---

## 9. P3：memory-preserving step repair

### 9.1 目的

P3 要实现 v6.18 未实现的 forward/update/launch fastpath，把 step 拉回 S2 并尝试 S1。

### 9.2 Candidate packages

```text
T0-GT4-canonical-baseline

T1-forward-fastpath-real:
  optimize forward grouped path, avoid redundant residual scale / grouped mix ops

T2-update-fastpath:
  reduce update prep and optimizer update overhead

T3-launch-fused-step:
  reduce forward/backward/update call boundary overhead

T4-layout-fastpath:
  remove contiguous/layout conversion if present

T5-kernel-count-reduction:
  fuse small post-kernel elementwise ops

T6-step-combo-safe:
  T1 + T2 + T3 + T4 + T5

T7-backward-no-recompute:
  explicitly assert and benchmark no hidden recompute path
```

### 9.3 必须记录字段

```text
package
implementation_status
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
step_improvement_vs_GT4
memory_regression_vs_GT4
forward_ms
backward_ms
update_ms
kernel_launch_overhead_proxy_ms
layout_conversion_ms
torch_op_count
triton_kernel_count
allocation_proxy_count
grad_relerr_max
grad_cos_min
```

### 9.4 判断标准

Step useful:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GT4}}}\leq0.96.
$$

Memory-preserving:

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq1.02.
$$

S2/S1 gates same as P2.

If step improves but memory exceeds $1.05$, it is runtime diagnostic only.

### 9.5 可视化

```text
p3_step_repair_pareto.svg
p3_step_time_waterfall.svg
p3_forward_update_launch_breakdown.svg
p3_memory_regression_bar.svg
```

---

## 10. P4：safe combo selection by survivor type

### 10.1 目的

P4 组合 P2/P3 单因素 repair。候选选择必须按 survivor type 和 Pareto，不得按 best memory。

### 10.2 Candidate packages

```text
C0-GT4-canonical-baseline
C1-best-step-preserving-memory-repair
C2-best-memory-preserving-step-repair
C3-safe-memory+step-combo
C4-S2-restore-combo
C5-S1-candidate-combo
C6-aggressive-S0-diagnostic
```

### 10.3 必须记录字段

```text
package
components
memory_ratio_min
memory_ratio_mean
memory_ratio_max
step_ratio_min
step_ratio_mean
step_ratio_max
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
memory_improvement_vs_GT4
step_improvement_vs_GT4
shape_stability_memory
shape_stability_step
worst_shape_memory_ratio
worst_shape_step_ratio
grad_relerr_max
grad_cos_min
survivor_type
open_one_step_probe
open_diagnostic_task
open_official_task
```

### 10.4 Survivor type

```text
S0:
  r_mem < 1.0 and r_step <= 1.20

S1:
  r_mem < 1.0 and r_step <= 1.35

S2:
  r_mem <= 1.05 and r_step <= 1.50

S3:
  memory better but step fail

S4:
  step better but memory fail

S5:
  gradient fail

S6:
  no improvement
```

### 10.5 打开规则

```text
S0/S1:
  open P5 one-step
  open P8 official task after P5 pass

S2:
  open P5 one-step
  open P6 diagnostic task after P5 pass

S3/S4/S5/S6:
  no task
```

### 10.6 可视化

```text
p4_combo_candidate_scorecard.svg
p4_s0_s1_s2_boundary_plot.svg
p4_shape_stability_heatmap.svg
p4_survivor_selection_dashboard.svg
```

---

## 11. P5：one-step probe if S2/S1/S0

### 11.1 目的

P5 只对 P4 S0/S1/S2 运行，验证 descent 和 rollback。没有 S2 不运行。

### 11.2 方法

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

### 11.3 必须记录

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
grad_norm
residual_ablation_delta_loss
residual_ablation_delta_logit
```

### 11.4 判断标准

P5 pass:

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

### 11.5 可视化

```text
p5_loss_before_after.svg
p5_bad_step_heatmap.svg
p5_update_norm_vs_loss_delta.svg
p5_residual_ablation_probe.svg
```

---

## 12. P6：diagnostic task gap attribution if S2 restored

### 12.1 目的

如果 S2 恢复，P6 重新打开 diagnostic task，继续分析 v6.16 的 task gap。如果 S2 没恢复，P6 仍为 `not_run`。

### 12.2 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-reference + ManualAdanLite
Best-v619-S2 + ManualAdanLite
Best-v619-S2 + ManualAdamW
```

如果 S1/S0 exists:

```text
Best-v619-S1 + ManualAdanLite
Best-v619-S1 + ManualAdamW
```

### 12.3 训练设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  120 diagnostic steps

logging:
  every 20 steps
```

### 12.4 必须记录字段

#### Curves

```text
train_loss_curve
val_loss_curve
train_acc_curve
val_acc_curve
test_acc_at_end
val_acc_at_end
ECE_curve
NLL_curve
```

#### Representation

```text
feature_effective_rank
feature_rank_ratio_vs_MLP
class_centroid_separation
within_class_variance
margin_mean
margin_p10
margin_p50
classwise_acc
hard_class_pairs
logit_norm_mean
logit_norm_std
```

#### Optimization

```text
optimizer_name
learning_rate
weight_decay
grad_norm
update_norm
update_over_param_norm
cos_update_grad
cos_update_prev
role_update_share_grouped_mix
role_update_share_residual
role_grad_norm_grouped_mix
role_grad_norm_residual
```

#### Efficiency

```text
task_step_time_ms
task_backward_memory_ratio
task_forward_ratio
task_backward_ratio
samples_per_second
time_to_val_acc_50
time_to_val_acc_60
time_to_val_acc_65
```

### 12.5 判断标准

Diagnostic task improvement pass:

$$
\operatorname{Acc}_{val,new}
\geq
\operatorname{Acc}_{val,GT4-v616}+0.02.
$$

Task gap is expressivity gap if:

$$
\operatorname{rank}_{KAN}<0.85\operatorname{rank}_{MLP}
$$

or:

$$
\operatorname{margin}_{p10,KAN}<0.85\operatorname{margin}_{p10,MLP}.
$$

Task gap is optimization gap if:

$$
\operatorname{ValLossAUC}_{step,KAN}
>
\operatorname{ValLossAUC}_{step,MLP}
$$

and update statistics show unstable or too-small updates.

Task gap is generalization gap if:

$$
(\operatorname{Acc}_{train,KAN}-\operatorname{Acc}_{val,KAN})
>
(\operatorname{Acc}_{train,MLP}-\operatorname{Acc}_{val,MLP})+0.03.
$$

### 12.6 可视化

```text
p6_loss_acc_curves_by_step.svg
p6_loss_acc_curves_by_time.svg
p6_feature_rank_vs_acc.svg
p6_margin_distribution.svg
p6_classwise_acc_heatmap.svg
p6_update_norm_trajectory.svg
p6_task_gap_taxonomy_dashboard.svg
```

---

## 13. P7：small optimizer / regularization diagnostic under S2

### 13.1 目的

P7 只在 S2 恢复后运行。它不是系统 optimizer exploration，而是小范围诊断。

### 13.2 Candidate diagnostics

```text
D0-best-S2-baseline-ManualAdanLite
D1-best-S2-ManualAdamW
D2-best-S2-AdanLite-lr-low
D3-best-S2-AdanLite-lr-high
D4-best-S2-AdamW-weightdecay-low
D5-best-S2-gradclip
D6-residual-scale-warmup
D7-crossgroup-lite-r1-edge-owned, if implemented
D8-group-shuffle-static
D9-label-smoothing-diagnostic
```

### 13.3 必须记录字段

```text
candidate
repair_type
nonKAN_param_count
edge_param_count_delta
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
val_acc
test_acc
ECE
NLL
train_val_gap
feature_rank
margin_p10
grad_norm_mean
update_norm_mean
cos_update_grad
bad_step_rate
```

### 13.4 判断标准

Efficiency envelope:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Diagnostic useful:

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.16 GT4+ManualAdanLite baseline or v6.19 best-S2 baseline.

If a repair improves accuracy but violates S2, it is diagnostic-only and cannot enter official route.

### 13.5 可视化

```text
p7_optimizer_expressivity_pareto.svg
p7_accuracy_vs_efficiency.svg
p7_feature_rank_improvement_bar.svg
p7_ece_train_val_gap_bar.svg
```

---

## 14. P8：official task re-entry if S1/S0

### 14.1 打开条件

Official task opens only if:

```text
survivor_type in {S0, S1}
P5 pass
r_mem < 1.0
r_step <= 1.35
grad pass
no fake/proxy
```

Diagnostic task opens if:

```text
survivor_type == S2
P5 pass
r_mem <= 1.05
r_step <= 1.50
```

### 14.2 Official task methods

If official task opens:

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
  Best-v619-S1 + ManualAdamW
  Best-v619-S1 + ManualAdanLite
  Best-v619-S1 + best diagnostic optimizer
```

### 14.3 必须记录

```text
train_loss_curve
val_loss_curve
test_acc
val_acc
train_acc
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
classwise_acc
seedwise_failure_reason
```

### 14.4 判断标准

Official task pass:

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

### 14.5 可视化

```text
p8_val_loss_vs_step.svg
p8_val_loss_vs_time.svg
p8_accuracy_vs_time.svg
p8_time_to_target_bar.svg
p8_task_efficiency_pareto.svg
p8_seedwise_delta_plot.svg
```

---

## 15. P9：optimizer exploration remains gated unless official task passes

P9 only opens after official P8 task pass. It is not v6.19 main stage.

If opened:

```text
ManualAdamW
ManualAdanLite
ManualWinLite
Lookahead-ManualAdamW
WarmupCosine-ManualAdamW
Best-diagnostic-recipe-from-P7
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

## 16. P10：functional correction remains gated

P10 only opens after P8 official task pass and P9 optimizer stability. Default:

```text
not_run
```

If opened, smoke only:

```text
task-only best
task + small residual geometry correction
task + low-frequency LightSmooth
```

Pass standards:

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

## 17. P11：route decision

### Route cases

```text
R1-S2Restored:
  v6.19 restores stable S2 and P5 pass.
  Open diagnostic task.

R2-S1Solved:
  v6.19 reaches S1/S0 and P5 pass.
  Open official task.

R3-RecomputeRejected:
  E1/recompute remains memory-good but step-bad.
  Exclude recompute from mainline.

R4-ProtocolOverheadResolved:
  P0/P2/P3 protocol mismatch identified and fixed.
  Continue kernel margin repair.

R5-ProtocolOverheadUnresolved:
  P0/P2/P3 remain inconsistent.
  Do not open task; fix measurement protocol.

R6-S2NotRestored:
  no S2 candidate.
  No task; continue kernel margin repair.

R7-S2RestoredTaskImproves:
  diagnostic task improves by >=2%.
  Continue fused grouped mainline.

R8-S2RestoredTaskGapPersists:
  S2 restored but task gap remains >5%.
  Need expressivity redesign.

R9-S1OfficialTaskPass:
  official task passes.
  Move to confirm seeds.

R10-S1OfficialTaskFail:
  efficiency solved but task fails.
  Focus expressivity/optimizer.
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
memory_improvement_vs_GT4
step_improvement_vs_GT4
survivor_type
one_step_probe_pass
official_task_opened
diagnostic_task_opened
best_val_acc
best_test_acc
val_acc_gap_vs_MLP
test_acc_gap_vs_MLP
task_gap_taxonomy
baseline_step_drift_resolved
canonical_protocol_pass
recompute_continuation_pass
open_optimizer_exploration
open_functional_correction
primary_blocker
next_required_implementation
no_fake
no_proxy
```

---

## 18. P12：artifact and failure audit

### Failure types

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_baseline_step_drift
F5_protocol_overhead_unresolved
F6_recompute_step_regression
F7_memory_repair_step_regression
F8_step_repair_memory_regression
F9_efficiency_margin_fail
F10_task_gap_expressivity
F11_task_gap_optimization
F12_task_gap_generalization
F13_task_gap_unclear
F14_official_task_gated
F15_diagnostic_task_gated
F16_optimizer_gated
F17_functional_gated
F18_fake_or_proxy_violation
F19_artifact_missing
```

### Artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p0_runner_kernel_matrix.csv
p1_protocol_overhead_attribution.csv
p1_component_margin_audit.csv
p2_nonrecompute_memory_repair.csv
p2_nonrecompute_memory_repair_detail.csv
p3_step_repair.csv
p3_step_repair_detail.csv
p4_combo_repair_selection.csv
p5_one_step_probe.csv
p6_diagnostic_task_gap_attribution.csv
p6_task_trace.csv
p7_optimizer_expressivity_diagnostic.csv
p8_task_reentry.csv
p8_task_trace.csv
p9_optimizer_exploration.csv
p10_functional_correction_smoke.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### Figures

```text
figures/p0_protocol_step_ratio_matrix.svg
figures/p0_step_drift_waterfall.svg
figures/p1_protocol_overhead_waterfall.svg
figures/p1_gt4_step_margin_waterfall.svg
figures/p2_memory_repair_pareto.svg
figures/p3_step_repair_pareto.svg
figures/p4_s0_s1_s2_boundary_plot.svg
figures/p5_loss_before_after.svg
figures/p6_loss_acc_curves_by_step.svg
figures/p6_loss_acc_curves_by_time.svg
figures/p6_task_gap_taxonomy_dashboard.svg
figures/p7_accuracy_vs_efficiency.svg
figures/p11_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 19. 成功与失败解释规则

### Case A：canonical protocol 统一后 S2 恢复

如果 GT4 baseline 或 safe combo 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

则恢复 S2。P5 one-step 必须打开。

### Case B：E1 仍 memory-good / step-bad

如果 E1 继续满足：

$$
M_{\text{E1}}<M_{\text{GT4}},
$$

但：

$$
T_{\text{step,E1}}>T_{\text{step,GT4}},
$$

则 E1 停止作为 mainline，只保留 diagnostic。

### Case C：step-preserving memory repair 成功

如果 M1-M8 中某个 non-recompute repair 同时降低 memory 且不伤 step，则进入 P4 combo。

### Case D：memory-preserving step repair 成功

如果 T1-T7 中某个 repair 降低 step 且不伤 memory，则进入 P4 combo。

### Case E：S1/S0 达成

如果：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

并且 P5 pass，则 official task 打开。

### Case F：S2 恢复但 task gap 仍大

如果 diagnostic task gap 仍大于 $5\%$，下一轮主线应转向 cross-group expressivity / optimizer diagnostic，而不是继续只修 memory。

### Case G：S2 无法恢复

如果 S2 仍无法恢复，则不允许 optimizer / task，下一轮继续 kernel/protocol margin repair或重新定义 canonical measurement。

---

## 20. 最终建议

v6.19 的一句话策略是：

$$
\boxed{
\text{停止 hidden-cache recompute，恢复 GT4 S2，并实现 non-recompute kernel margin repair。}
}
$$

v6.18 的主要教训是：memory repair 必须是 step-preserving 的。下一步应优先实现：

```text
grad_mix / grad_poly buffer reuse
update_temp reuse
short-lived grouped output free
triton scratch lifetime trim
forward/update/launch fastpath
```

只有 S2 恢复后，才重新打开 one-step 和 diagnostic task；只有 S1/S0 达成后，才进入 official task 与系统 optimizer exploration。
