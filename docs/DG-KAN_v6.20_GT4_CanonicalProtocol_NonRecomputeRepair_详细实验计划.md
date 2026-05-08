# DG-KAN v6.20：Canonical Protocol Lock、GT4 S2 稳定恢复、Non-Recompute Margin Repair 与 Task Gap Reopen 详细实验计划

> 本计划基于 v6.19 final real-only run 制定。v6.19 的最终 route 是 `R5-BaselineStepDriftUnresolved`。这不是 fused grouped 路线失败，而是 **canonical timing protocol 尚未闭合 + GT4 S2 step margin 不稳定 + non-recompute kernel repair 未实现**。本轮最接近成功的地方是：P0 中多个 GT4 baseline 已接近或达到 S2 step gate 附近，memory ratio 稳定为 `1.0312`；但 final canonical protocol 仍失败，P2/P3 repair protocol 下 baseline step ratio 仍高于 `1.50`，E1 recompute 虽然降低 memory 到 `1.0234`，但 step 恶化到 `1.6009`。因此 v6.20 的目标是：锁定 canonical protocol，恢复稳定 S2，再做 non-recompute kernel margin repair，并在 S2 后重新打开 diagnostic task gap attribution。

---

## 0. 实验整体目标

v6.20 的整体目标不是回到 DWM2，不是重新大搜 low-rank / piecewise，也不是现在就大规模优化 optimizer。v6.20 的核心目标是围绕 fused grouped GT4 完成一次严谨的工程闭环：

$$
\boxed{
\text{在统一 timing protocol 下稳定恢复 GT4 S2，并用 non-recompute repair 推向 S1。}
}
$$

更具体地说，v6.20 要回答四个问题。

第一个问题：

$$
\boxed{
\text{v6.19 的 P0、P2、P3 为什么对同一 GT4 baseline 给出不同 step ratio？}
}
$$

第二个问题：

$$
\boxed{
\text{在 canonical protocol 下，GT4 baseline 是否稳定满足 S2？}
}
$$

第三个问题：

$$
\boxed{
\text{能否不使用 hidden-cache recompute，通过 buffer lifetime / update reuse / launch fastpath 修复 S1 margin？}
}
$$

第四个问题：

$$
\boxed{
\text{S2 恢复后，diagnostic task gap 是否来自 optimizer、表达力、泛化，还是短训练预算？}
}
$$

v6.19 的关键事实如下：

```text
P0 A2-GT4-v616-runner-reference:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5003
  backward ratio mean = 0.7709
  v6.16 repro pass    = 0 by strict delta

P0 A3-E0-GT4-v617-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4969
  backward ratio mean = 0.7696

P0 A5-GT4-v616-kernel-with-v617-runner:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4965
  backward ratio mean = 0.7669

P0 A6-GT4-v617-kernel-with-v616-timing-protocol:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4979
  backward ratio mean = 0.7691

P2 M0-GT4-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5319
  backward ratio mean = 0.7946

P3 T0-GT4-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5238
  backward ratio mean = 0.7753

P2 M8-E1-recompute-hidden-cache:
  memory ratio mean   = 1.0234
  step ratio mean     = 1.6009
  backward ratio mean = 0.9510
```

v6.19 的 route 是：

```text
R5-BaselineStepDriftUnresolved
primary_blocker = baseline_step_drift
next_required_implementation = fix_runner_protocol_before_task
```

这意味着 v6.20 的最低目标必须先恢复 S2：

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

v6.20 的主要工程目标是达到 S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

如果 S1 暂时不能达到，但 S2 稳定恢复且 diagnostic task 明显改善，则 fused grouped 仍继续作为主线。Diagnostic task improvement 定义为：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.16 `GT4+ManualAdanLite` baseline or v6.20 restored-S2 baseline.

---

## 1. 当前实验进展与问题判断

### 1.1 已经确认的正向事实

v6.19 不是原地失败。它确认了几个重要事实。

第一，GT4 memory baseline 非常稳定。P0/P1/P2/P3 中 GT4 相关 baseline 的 mean memory ratio 都保持在 `1.0312`，absolute peak allocated mean 为 `18.9276 MB`。这说明 fused grouped 的 memory path 没有退化。

第二，P0 中多个 GT4 baseline 的 step ratio 已经非常接近 S2：A3/A5/A6 分别约为 `1.4969 / 1.4965 / 1.4979`。这说明 GT4 在某些 protocol 下已经处于 S2 边界内或边界附近。

第三，E1 recompute 的 memory 改善真实存在：memory ratio 从 `1.0312` 到 `1.0234`，absolute peak 从 `18.9276 MB` 到 `18.7817 MB`。但这个 memory 改善很小，且 step/backward 代价明显。

第四，本轮没有 fake/proxy，没有 gradient correctness failure，没有 artifact missing。P5-P10 gate 是正确的，不是漏跑。

### 1.2 当前失败的本质

v6.19 的失败本质是：

$$
\boxed{
\text{GT4 的 S2 margin 太薄，且 canonical timing protocol 没有闭合。}
}
$$

P0 显示 GT4 已接近 S2，但 P2/P3 repair protocol 又让 baseline step 变成 `1.5238-1.5319`，高于 `1.50`。final route 因此保守写为 `R5-BaselineStepDriftUnresolved`。

这不是 optimizer 问题，也不是 task 问题，因为没有 S2，就不应该进入 P5 one-step 和 P6 diagnostic task。

### 1.3 当前错误方向已经清楚：hidden-cache recompute

E1 的模式非常明确：

$$
M_{\text{E1}}<M_{\text{GT4}},
$$

但：

$$
T_{\text{step,E1}}>T_{\text{step,GT4}},
$$

并且：

$$
T_{\text{backward,E1}}>T_{\text{backward,GT4}}.
$$

它属于 memory-better / step-worse 的 S3 diagnostic。v6.20 不应继续把 E1 放进主 combo。它只作为负例和 memory diagnostic 保留。

### 1.4 当前是否接近目标

答案是：

$$
\boxed{
\text{接近 S2，非常接近；但离 S1 official gate 仍有明确工程距离。}
}
$$

如果以 P0 A5/A6 计算，GT4 已经几乎恢复 S2：

```text
A5 step ratio = 1.4965
A6 step ratio = 1.4979
```

它们只需要保持稳定即可满足 S2。但 S1 要求：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

以 memory ratio `1.0312` 计算，到 S1 memory gate 还需要：

$$
1-\frac{1.00}{1.0312}\approx3.03\%.
$$

以 P0 A6 step `1.4979` 计算，到 S1 step gate 还需要：

$$
1-\frac{1.35}{1.4979}\approx9.87\%.
$$

以 P2 M0 step `1.5319` 计算，到 S2 step gate 只差：

$$
1-\frac{1.50}{1.5319}\approx2.08\%.
$$

到 S1 step gate 还需要：

$$
1-\frac{1.35}{1.5319}\approx11.87\%.
$$

所以现在不是大方向失败，而是：

```text
S2: 几乎到了，但 protocol 不稳。
S1: 还需要约 3% memory reduction 和约 10%-12% step speedup。
Task: 还没资格在 v6.19 更新判断。
```

---

## 2. v6.20 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。v6.20 不允许新增 DWM2 patch。

第三，low-rank、piecewise、Python grouped loop 不作为主线。它们只能作为历史 reference，不进入 route selection。

第四，E1 hidden-cache recompute 不允许进入 main combo。它只能作为 diagnostic negative control，除非它同时满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

第五，不允许按 best memory 选择 candidate。v6.17-v6.19 已反复证明 best-memory E1 是 S3。v6.20 必须按 survivor type 和 Pareto 选择。

第六，没有 S2 时，不允许打开 one-step、diagnostic task、official task、optimizer exploration、functional correction。

第七，S2 只允许 diagnostic task，不允许 official success claim。Official task 必须等 S1/S0。

第八，optimizer / regularization diagnostic 只能在 S2 恢复后运行。如果没有 S2，optimizer 结果会被 efficiency gate 混淆。

第九，所有 repair 必须记录 absolute memory MB，不允许只报告 ratio。

第十，任何 task-gap 修复不能引入 non-KAN trainable parameters。所有 cross-group / scale / schedule / regularization 诊断必须保持 strict PureKAN accounting。

---

## 3. 核心假设

### H1：v6.19 的主要 blocker 是 canonical timing protocol 未统一，而不是 GT4 kernel 失效

P0 中 A3/A5/A6 已经接近或满足 S2，但 P2/P3 protocol 的 baseline step 又超 gate。H1 假设：差异来自 measurement protocol / wrapper overhead / component logging / timing sync / repair package dispatch，而不是 GT4 fused kernel 本身退化。

H1 成立要求：

```text
P0/P2/P3 使用同一 canonical protocol 后，GT4 baseline step delta <= 0.02
```

即：

$$
|r_{\text{step,P0-GT4}}-r_{\text{step,P2-M0}}|\leq0.02,
$$

$$
|r_{\text{step,P0-GT4}}-r_{\text{step,P3-T0}}|\leq0.02.
$$

并且 restored GT4 baseline 满足：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果 H1 成立，则可以认为 fused grouped GT4 底座仍有效，继续做 kernel margin repair。

---

### H2：E1 hidden-cache recompute 是稳定的 memory-time tradeoff，不应进入主 combo

H2 假设 E1 的收益/代价模式稳定存在：

```text
memory better
backward worse
step worse
```

H2 成立标准：

$$
\frac{M_{\text{E1}}}{M_{\text{GT4}}}<1.00,
$$

但：

$$
\frac{T_{\text{step,E1}}}{T_{\text{step,GT4}}}>1.02.
$$

如果成立，则 E1 只作为 diagnostic negative control，不进入 P4 safe combo。

---

### H3：S1 memory gap 应通过 non-recompute buffer lifetime repair 修复

GT4 memory gap 到 S1 只有约 $3\%$。H3 假设合适的修复对象不是 hidden cache，而是：

```text
grad_mix buffer
grad_poly buffer
update_temp
triton_scratch
short-lived grouped output
reserved / allocated gap
```

H3 的单项 repair 成立标准：

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985,
$$

且：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GT4}}}\leq1.02.
$$

也就是 memory 至少降低 $1.5\%$，step 不恶化超过 $2\%$。

---

### H4：S1 step gap 应通过 forward/update/launch fastpath 修复，而不是继续优化 backward

GT4 的 backward 已经低于 MLP，step gap 更可能来自：

```text
forward path
update prep
optimizer update
kernel launch overhead
torch wrapper overhead
layout conversion
component logging overhead
```

H4 的单项 repair 成立标准：

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GT4}}}\leq0.96,
$$

且：

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq1.02.
$$

也就是 step 至少提速 $4\%$，memory 不恶化超过 $2\%$。

---

### H5：S2 恢复后才能重新判断 task gap

v6.16 diagnostic task gap 仍是重要问题：

```text
GT4+ManualAdanLite val acc ≈ 0.6302
MLP-autograd val acc ≈ 0.6981
gap ≈ -6.79 percentage points
```

但 v6.19 没有 task run，因此不能更新这个判断。H5 要求：

```text
If S2 restored:
  run one-step probe
  if one-step pass:
    run diagnostic task gap attribution

If no S2:
  no task
  no optimizer
```

Diagnostic task 不能只记录 accuracy，必须记录：

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

---

### H6：如果 v6.20 统一 protocol 后仍无法稳定 S2，则需要优先修 measurement/runner，而不是换 optimizer

H6 是 Stop/Go 假设。如果完成：

```text
canonical protocol lock
protocol overhead attribution
GT4 baseline retest
non-recompute repair
step repair
```

仍没有 S2，那么不能开 optimizer，不能开 task。下一轮必须继续修 runner/protocol/kernel margin，或者明确重新定义 canonical measurement gate。

---

## 4. 实验阶段总览

v6.20 分为十三个阶段：

```text
P0: canonical protocol lock and GT4 baseline retest
P1: protocol overhead and margin attribution
P2: non-recompute memory repair implementation
P3: step fastpath repair implementation
P4: safe combo selection by survivor type
P5: one-step probe if S2/S1/S0
P6: diagnostic task gap attribution if S2 restored
P7: small optimizer / regularization diagnostic under S2
P8: cross-group / expressivity diagnostic under S2
P9: official task re-entry if S1/S0
P10: optimizer exploration remains gated unless official task passes
P11: functional correction remains gated
P12: route decision
P13: artifact and failure audit
```

P0-P4 是硬核心。P5-P8 只有 S2 以上才运行。P9-P11 默认关闭。

---

## 5. P0：canonical protocol lock and GT4 baseline retest

### 5.1 目的

P0 的目标是锁定 canonical timing protocol。v6.20 不能再出现 P0 接近 S2、P2/P3 又超 gate 且无法解释的情况。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-runner-reference
GT4-v617-baseline
GT4-v618-baseline
GT4-v619-P0-baseline
P2-M0-GT4-baseline-reproduced
P3-T0-GT4-baseline-reproduced
E1-recompute-hidden-cache-diagnostic
```

### 5.3 Protocol variants

必须显式比较：

```text
protocol_A_v616_timing
protocol_B_v618_p0_side_by_side
protocol_C_v619_repair_protocol
protocol_D_new_canonical_minimal_logging
protocol_E_new_canonical_with_component_logging
```

每个 protocol 必须记录：

```text
warmup_steps
measure_steps
cuda_sync_policy
component_logging_enabled
wandb_logging_inside_loop
memory_history_enabled
profiler_context_enabled
repair_dispatch_enabled
torch_op_count_enabled
allocation_proxy_enabled
```

### 5.4 必须记录字段

```text
variant
protocol
dataset
batch_size
depth
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
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
kernel_launch_proxy_count
warmup_steps
measure_steps
canonical_protocol_pass
s2_pass
```

### 5.5 判断标准

Canonical protocol pass requires:

$$
\max_{p,q}
|r_{\text{step,GT4}}^{(p)}-r_{\text{step,GT4}}^{(q)}|
\leq0.02
$$

across accepted canonical protocols.

GT4 S2 restoration pass:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

If no protocol can produce stable GT4 S2 and protocol variation remains larger than $0.02$, route must be:

```text
R5-ProtocolOverheadUnresolved
```

### 5.6 可视化

```text
p0_protocol_step_ratio_matrix.svg
p0_protocol_memory_ratio_matrix.svg
p0_protocol_delta_heatmap.svg
p0_s2_pass_by_protocol.svg
p0_canonical_protocol_dashboard.svg
```

---

## 6. P1：protocol overhead and margin attribution

### 6.1 目的

P1 分解 protocol overhead 与 GT4 margin。它要回答：P2/P3 repair protocol 为什么比 P0 慢。

### 6.2 测量对象

```text
GT4-canonical-minimal
GT4-canonical-with-component-logging
GT4-repair-dispatch-wrapper
GT4-memory-history-enabled
GT4-wandb-loop-disabled
GT4-wandb-loop-enabled
E1-recompute-hidden-cache
MLP-manual-linear-reference
```

### 6.3 Component phases

```text
phase_forward_fused_grouped
phase_forward_residual_scale
phase_forward_grouped_mix
phase_loss_delta
phase_backward_fused_grouped
phase_backward_dx
phase_backward_grad_mix
phase_backward_grad_poly
phase_update_prep
phase_optimizer_update
phase_timing_sync
phase_component_logging
phase_repair_dispatch
phase_wandb_logging
phase_memory_history
phase_cleanup
```

### 6.4 必须记录字段

```text
variant
protocol
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

E1-specific:

```text
hidden_cache_saved_MB
recompute_time_ms
backward_extra_time_ms_vs_GT4
memory_saved_per_extra_ms
recompute_kernel_count
recompute_torch_op_count
```

### 6.5 判断标准

P1 pass requires:

```text
primary_protocol_overhead_source identified
unknown_time_fraction <= 0.10
unknown_memory_fraction <= 0.10
```

E1 rejection criterion:

$$
\frac{T_{\text{step,E1}}}{T_{\text{step,GT4}}}>1.02.
$$

If E1 rejection criterion holds, E1 cannot enter P4 combo.

### 6.6 可视化

```text
p1_protocol_overhead_waterfall.svg
p1_gt4_step_margin_waterfall.svg
p1_gt4_memory_margin_waterfall.svg
p1_e1_memory_saved_per_extra_ms.svg
p1_phase_time_stacked_bar.svg
```

---

## 7. P2：non-recompute memory repair implementation

### 7.1 目的

P2 实现真正的 non-recompute memory repair。v6.19 的 M1-M7 仍未实现，v6.20 必须至少真实实现两个。

### 7.2 Candidate packages

```text
M0-GT4-canonical-baseline

M1-grad-mix-buffer-reuse:
  reuse grad_mix buffer after accumulation, no recompute

M2-grad-poly-buffer-reuse:
  reuse grad_poly buffer and avoid duplicate grad temp

M3-update-temp-reuse:
  reuse update temp and avoid separate update materialization

M4-triton-scratch-lifetime-trim:
  release or reuse Triton scratch earlier

M5-reserved-memory-trim:
  reduce reserved-unallocated gap and allocator padding

M6-short-lived-output-free:
  shorten grouped output lifetime after backward consumes it

M7-grad-update-buffer-combo:
  M1 + M2 + M3

M8-safe-memory-combo:
  M1 + M2 + M3 + M4 + M6

M9-E1-recompute-hidden-cache:
  diagnostic only
```

### 7.3 必须记录字段

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

### 7.4 判断标准

Memory useful:

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985.
$$

Step-preserving:

$$
\frac{T_{\text{step,new}}}{T_{\text{step,GT4}}}\leq1.02.
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

### 7.5 可视化

```text
p2_memory_repair_pareto.svg
p2_memory_source_reduction_bar.svg
p2_step_preservation_bar.svg
p2_recompute_vs_nonrecompute_tradeoff.svg
```

---

## 8. P3：step fastpath repair implementation

### 8.1 目的

P3 实现真实 step repair。v6.19 的 T1-T6 仍未实现，v6.20 至少实现一个 forward/update/launch fastpath。

### 8.2 Candidate packages

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
  assert no hidden recompute path
```

### 8.3 必须记录字段

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

### 8.4 判断标准

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

### 8.5 可视化

```text
p3_step_repair_pareto.svg
p3_step_time_waterfall.svg
p3_forward_update_launch_breakdown.svg
p3_memory_regression_bar.svg
```

---

## 9. P4：safe combo selection by survivor type

### 9.1 目的

P4 组合 P2/P3 单因素 repair。候选选择必须按 survivor type 和 Pareto，不得按 best memory。

### 9.2 Candidate packages

```text
C0-GT4-canonical-baseline
C1-best-step-preserving-memory-repair
C2-best-memory-preserving-step-repair
C3-safe-memory+step-combo
C4-S2-restore-combo
C5-S1-candidate-combo
C6-aggressive-S0-diagnostic
```

### 9.3 必须记录字段

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

### 9.4 Survivor type

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

### 9.5 打开规则

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

### 9.6 可视化

```text
p4_combo_candidate_scorecard.svg
p4_s0_s1_s2_boundary_plot.svg
p4_shape_stability_heatmap.svg
p4_survivor_selection_dashboard.svg
```

---

## 10. P5：one-step probe if S2/S1/S0

### 10.1 目的

P5 只对 P4 S0/S1/S2 运行。没有 S2 不运行。

### 10.2 方法

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

### 10.3 必须记录

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

### 10.4 判断标准

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

### 10.5 可视化

```text
p5_loss_before_after.svg
p5_bad_step_heatmap.svg
p5_update_norm_vs_loss_delta.svg
p5_residual_ablation_probe.svg
```

---

## 11. P6：diagnostic task gap attribution if S2 restored

### 11.1 目的

如果 S2 恢复，P6 重新打开 diagnostic task，继续分析 v6.16 的 task gap。如果 S2 没恢复，P6 仍为 `not_run`。

### 11.2 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-reference + ManualAdanLite
Best-v620-S2 + ManualAdanLite
Best-v620-S2 + ManualAdamW
```

如果 S1/S0 exists:

```text
Best-v620-S1 + ManualAdanLite
Best-v620-S1 + ManualAdamW
```

### 11.3 训练设置

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

### 11.4 必须记录字段

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

### 11.5 判断标准

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

### 11.6 可视化

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

## 12. P7：small optimizer / regularization diagnostic under S2

### 12.1 目的

P7 只在 S2 恢复后运行。它不是系统 optimizer exploration，而是小范围诊断。

### 12.2 Candidate diagnostics

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

### 12.3 必须记录字段

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

### 12.4 判断标准

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

relative to v6.16 GT4+ManualAdanLite baseline or v6.20 best-S2 baseline.

If a repair improves accuracy but violates S2, it is diagnostic-only and cannot enter official route.

### 12.5 可视化

```text
p7_optimizer_expressivity_pareto.svg
p7_accuracy_vs_efficiency.svg
p7_feature_rank_improvement_bar.svg
p7_ece_train_val_gap_bar.svg
```

---

## 13. P8：official task re-entry if S1/S0

### 13.1 打开条件

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

### 13.2 Official task methods

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
  Best-v620-S1 + ManualAdamW
  Best-v620-S1 + ManualAdanLite
  Best-v620-S1 + best diagnostic optimizer
```

### 13.3 必须记录

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

### 13.4 判断标准

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

## 14. P9：optimizer exploration remains gated unless official task passes

P9 only opens after official P8 task pass. It is not v6.20 main stage.

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

## 15. P10：functional correction remains gated

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

## 16. P11：route decision

### Route cases

```text
R1-S2Restored:
  v6.20 restores stable S2 and P5 pass.
  Open diagnostic task.

R2-S1Solved:
  v6.20 reaches S1/S0 and P5 pass.
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

## 17. P12：artifact and failure audit

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

## 18. 成功与失败解释规则

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

but:

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

## 19. 最终建议

v6.20 的一句话策略是：

$$
\boxed{
\text{先统一 measurement protocol，再用 non-recompute repair 恢复并稳定 GT4 S2。}
}
$$

最重要的不是再证明 E1 能省一点 memory，而是实现之前一直未完成的真实修复：

```text
grad_mix / grad_poly buffer reuse
update_temp reuse
short-lived grouped output free
triton scratch lifetime trim
forward/update/launch fastpath
```

只有 S2 恢复后，才重新打开 one-step 和 diagnostic task；只有 S1/S0 达成后，才打开 official task 与系统 optimizer exploration。
