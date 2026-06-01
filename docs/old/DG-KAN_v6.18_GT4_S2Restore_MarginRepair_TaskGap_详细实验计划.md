# DG-KAN v6.18：GT4 S2 恢复、Step-Preserving Margin Repair 与 Task Gap Reopen 详细实验计划

> 本计划基于 v6.17 final real-only run 制定。v6.17 的真实 route 是 `R5-EfficiencyMarginFailed`。本轮唯一新增真实 repair `E1-GT4-no-extra-output-cache / recompute-hidden-cache` 把 memory ratio mean 从 `1.0312` 降到 `1.0234`，但 step ratio 从 `1.5191` 进一步恶化到 `1.5891`，backward ratio 从 `0.7648` 恶化到 `0.9187`。因此 v6.17 没有 S0/S1/S2 candidate，P7 one-step、P8 task、P9 optimizer、P10 functional 全部正确 gate。v6.18 的目标不是扩大搜索，也不是马上做 optimizer 主线，而是恢复并稳定 v6.16 的 GT4 S2 baseline，在不牺牲 step 的前提下做 memory/step margin repair，并重新打开 diagnostic task gap attribution。

---

## 0. 实验整体目标

v6.18 的整体目标是回答三个问题。

第一个问题：

$$
\boxed{
\text{v6.17 为什么没有复现 v6.16 的 GT4 S2 step margin？}
}
$$

第二个问题：

$$
\boxed{
\text{能否在不使用 expensive recompute 的情况下，把 GT4 推回 S2，并进一步推到 S1/S0？}
}
$$

第三个问题：

$$
\boxed{
\text{当 S2 恢复后，diagnostic task gap 是表达力、优化器、泛化，还是训练预算问题？}
}
$$

v6.16 的有效 baseline 是：

```text
GT4-best-fused-no-materialize:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.4426
  backward ratio mean = 0.7281
  forward ratio mean  = 1.1425
  grad relerr max     = 1.92e-08
  survivor type       = S2
  one-step probe      = pass
  diagnostic task     = measured
```

v6.17 的有效结果是：

```text
E0-GT4-v616-baseline:
  memory ratio mean   = 1.0312
  step ratio mean     = 1.5191
  backward ratio mean = 0.7648
  forward ratio mean  = 1.2338
  S0/S1/S2            = 0/0/0

E1-GT4-no-extra-output-cache:
  memory ratio mean   = 1.0234
  step ratio mean     = 1.5891
  backward ratio mean = 0.9187
  forward ratio mean  = 1.1988
  S0/S1/S2            = 0/0/0
```

因此 v6.18 的最低目标不是追求新的 primitive family，而是恢复 S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

$$
\operatorname{grad\_relerr}<10^{-4},
$$

$$
\operatorname{grad\_cos}>0.999.
$$

v6.18 的主要工程目标是达到 S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

如果 S1 暂时没有达到，但恢复 S2 并且 diagnostic task 比 v6.16 GT4 明显改善，则 fused grouped 仍继续作为主线。Diagnostic task improvement 定义为：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to v6.16 `GT4+ManualAdanLite` baseline, and no efficiency regression beyond S2.

---

## 1. 当前实验进展与问题判断

### 1.1 已经成功建立的底座

从 v6.16 开始，fused grouped 路线已经不再是空想。真实 Triton fused grouped kernel 已经实现，梯度 relerr 达到 $10^{-8}$ 量级，并且 v6.16 的 GT4 达到 S2。这个结果说明：

```text
grouped memory/runtime tradeoff 可以被 fused kernel 打开
graph-free manual backward 仍然可靠
strict PureKAN accounting 仍然可保持
fused grouped 是当前最强主线
```

v6.17 没有推翻这个底座。v6.17 的失败是 margin repair 失败，不是 fused grouped 路线整体失败。

### 1.2 v6.17 的真实失败

v6.17 唯一新增真实 repair 是 `E1-GT4-no-extra-output-cache`，它采用 forward 不保存 hidden cache、backward 从原始 input 重算 layer input 的策略。该策略确实降低了 memory：

$$
1-\frac{1.0234}{1.0312}\approx0.76\%.
$$

但它让 step 明显变慢：

$$
\frac{1.5891}{1.5191}-1\approx4.61\%
$$

relative to v6.17 E0, and relative to v6.16 GT4:

$$
\frac{1.5891}{1.4426}-1\approx10.16\%.
$$

同时 backward ratio 从 v6.17 E0 的 `0.7648` 变为 `0.9187`。这说明 recompute hidden cache 是典型的 memory-time tradeoff：它省了一点 memory，但把已经很优秀的 backward path 变慢了。

### 1.3 v6.17 的另一个关键问题：baseline step drift

v6.16 GT4 的 step ratio 是 `1.4426`，但 v6.17 E0-GT4 baseline 的 step ratio 是 `1.5191`。这意味着 v6.17 即使不加 E1，也没有复现 S2 step gate。这个 drift 必须先解释，否则继续做 E2/E3/E4 很容易把 measurement drift 当成 repair failure。

因此 v6.18 的第一阶段必须做：

```text
v616 runner / v617 runner side-by-side reproduction
same seed
same grid
same warmup/measure
same Triton kernel
same data loader
same profiler
same CUDA sync protocol
same W&B off/on policy check
```

### 1.4 当前不应全面转向 optimizer

v6.17 没有 S2 candidate，因此 P8 task 没有打开，P9 optimizer 也没有打开。本轮不能分析 task gap，也不能说 optimizer 失败。v6.16 的 diagnostic task gap 仍然是有效参考，但 v6.18 必须先恢复 S2，才能重新打开 diagnostic task。否则 optimizer 结果会被 step/memory gate 混淆。

### 1.5 当前路线判断

当前仍在正确道路上，但要非常聚焦。下一步不应该：

```text
回到 DWM2
回到 low-rank / piecewise 主线
继续 recompute hidden cache
直接大规模 optimizer sweep
直接开 official task
```

正确路线是：

```text
恢复 v6.16 GT4 S2
解释 v6.17 step drift
做 step-preserving memory repair
做 memory-preserving step repair
恢复 diagnostic task gap attribution
在 S1/S0 后才打开 official task
```

---

## 2. v6.18 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或 derived rows。所有未实现内容必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，DWM2 继续冻结。v6.18 不允许新增 DWM2 patch。

第三，low-rank、piecewise、Python grouped loop 不再作为主线。它们只能作为 reference 或 fallback，不允许抢占 fused grouped 主线。

第四，不允许把 E1 当作 task candidate。E1 是 memory diagnostic，不是 S2 survivor。

第五，不允许再做 expensive hidden recompute 作为主修复，除非同时满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

第六，不允许只按 best memory 选择 candidate。v6.17 的 P3 选择 E1 是因为 best memory，但它 step 失败，最终变成 S3。v6.18 的 candidate selection 必须按 survivor type 优先，而不是按 memory 单指标。

第七，没有 S2 时，不允许打开：

```text
one-step probe
diagnostic task
official task
optimizer exploration
functional correction
```

第八，S2 可以打开 diagnostic task，但不能称为 official task success。Official task 必须等 S1/S0：

$$
r_{\text{mem}}<1.00,\quad r_{\text{step}}\leq1.35.
$$

第九，任何 task-gap 修复不能引入 non-KAN trainable parameters。Cross-group、scale、regularization、optimizer 诊断都必须保持：

```text
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
```

---

## 3. 核心假设

### H1：v6.17 的主要失败不是 fused grouped 失效，而是 E1 recompute 造成 step regression

H1 假设：E1 的 hidden recompute 减少了少量 cache memory，但增加了 backward recompute 和 launch/compute 开销。H1 成立的标准是：

$$
r_{\text{mem,E1}}<r_{\text{mem,E0}},
$$

但：

$$
r_{\text{step,E1}}>r_{\text{step,E0}},
$$

且：

$$
r_{\text{backward,E1}}>r_{\text{backward,E0}}.
$$

v6.17 已经满足这个模式。v6.18 要进一步验证 E1 的 step regression 是否稳定出现，并记录：

```text
recompute_time_ms
backward_recompute_layer_input_ms
extra_kernel_count
extra_torch_op_count
hidden_cache_saved_MB
memory_saved_per_extra_ms
```

如果每节省 $1$ MB memory 需要增加过多 step time，则 E1 不再进入 combo。

---

### H2：v6.17 baseline step drift 来自 measurement / runner / forward overhead，而不是 GT4 kernel 本身退化

v6.16 GT4 step ratio 是 `1.4426`，v6.17 E0 step ratio 是 `1.5191`。H2 假设：这个差异可能来自 runner/profiler protocol、measurement variance、task/gate wrapper、forward path overhead 或 CUDA timing方式，而不是 Triton kernel 本身发生了性能退化。

H2 成立标准是 side-by-side reproduction 中存在至少一个 setup 复现 v6.16 step：

$$
|r_{\text{step,repro}}-1.4426|\leq0.05.
$$

如果所有 setup 都只能得到约 `1.52`，则 v6.16 S2 需要重新校准，v6.18 应以新 baseline 为准。

---

### H3：GT4 离 S1/S0 的 memory gap 应优先用 buffer lifetime / update temp reuse 修，而不是 hidden recompute

GT4 memory gap 只有约 $3\%$。H3 假设：剩余 memory 来源可能是：

```text
grad_mix buffer
grad_poly buffer
update_temp
optimizer_state scratch
triton scratch
reserved/allocated gap
short-lived grouped output
```

这些比 hidden recompute 更适合作为 repair target。H3 成立要求 P1 memory attribution 至少找出一个可修 source，且 estimated saving 可以达到：

$$
\Delta r_{\text{mem}}\geq0.015.
$$

H3 repair 成立要求：

$$
\frac{M_{\text{new}}}{M_{\text{GT4}}}\leq0.985,
$$

and:

$$
\frac{T_{\text{step,new}}}{T_{\text{GT4}}}\leq1.02.
$$

即 memory 降低至少 $1.5\%$，step 不能显著变差。

---

### H4：GT4 的 step gap 主要来自 forward/update/launch，而不是 backward

v6.16 GT4 backward ratio 已经是 `0.7281`，v6.17 E0 backward ratio 是 `0.7648`，都低于 MLP。H4 假设：剩余 step gap来自：

```text
forward fused grouped path
optimizer update
update prep
kernel launch overhead
torch wrapper overhead
layout conversion
reserved memory trimming overhead
```

H4 成立要求 P1/P2 component timing 显示：

$$
\frac{T_{\text{forward+update+launch}}}{T_{\text{step overhead}}}\geq0.50.
$$

其中 step overhead 定义为：

$$
T_{\text{step overhead}}=T_{\text{step,GT4}}-1.35T_{\text{step,MLP}}.
$$

H4 repair 目标是：

$$
r_{\text{step,new}}\leq1.35.
$$

---

### H5：task gap 只有在 S2 恢复后才能诊断

v6.17 没有 S2，因此不能运行 task gap attribution。H5 假设：v6.16 diagnostic task gap 仍然是下一步的有效问题，但必须在 v6.18 先恢复 S2 后才继续诊断。

S2 恢复标准：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果恢复 S2，则 diagnostic task 必须记录：

```text
feature rank
margin
classwise acc
ECE
NLL
train-val gap
update stats
role update share
```

不能只报告 val/test acc。

---

### H6：如果 v6.18 不能恢复 S2，则 fused grouped margin repair 需要回到 v6.16 kernel/protocol，而不是扩展 optimizer

H6 是 Stop/Go 假设。如果 v6.18 完成：

```text
v616/v617 side-by-side reproduction
E0/E1 component timing
step-preserving memory repair
memory-preserving step repair
```

仍没有 S2，则不应打开 optimizer。路线应回到 kernel/protocol repair，重新确认 fused grouped kernel baseline。

---

## 4. 实验阶段总览

v6.18 分为十二个阶段：

```text
P0: v6.16/v6.17 side-by-side reproduction and contract
P1: GT4/E1 efficiency margin attribution
P2: step-preserving memory repair
P3: memory-preserving step repair
P4: combo repair and survivor selection
P5: one-step probe for S2/S1/S0
P6: diagnostic task gap attribution if S2 restored
P7: lightweight expressivity / optimizer diagnostic under S2 envelope
P8: official task re-entry gate if S1/S0
P9: optimizer exploration remains gated unless official task passes
P10: functional correction remains gated
P11: route decision
P12: artifact and failure audit
```

P0-P4 是本轮硬核心。P5-P7 只有 S2 以上才运行。P8-P10 默认关闭。

---

## 5. P0：v6.16/v6.17 side-by-side reproduction and contract

### 5.1 目的

确认 v6.17 的 step regression 是真实 kernel regression、runner drift，还是 measurement variance。P0 需要在同一套环境中重跑 v6.16 GT4 与 v6.17 E0/E1。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-runner-reference
E0-GT4-v617-baseline
E1-GT4-recompute-hidden-cache
GT4-v616-kernel-with-v617-runner
GT4-v617-kernel-with-v616-timing-protocol, if feasible
```

### 5.3 必须记录字段

```text
variant
runner_version
kernel_version
timing_protocol
dataset
batch_size
depth
status
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
kernel_count_total
torch_op_count
cuda_sync_count
warmup_steps
measure_steps
reproduction_delta_vs_v616_memory
reproduction_delta_vs_v616_step
```

### 5.4 判断标准

Contract pass:

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
uses_loss_backward = 0
manual_backward_available = 1
grad_relerr < 1e-4
```

v6.16 reproduction pass:

$$
|r_{\text{mem,repro}}-1.0312|\leq0.03.
$$

$$
|r_{\text{step,repro}}-1.4426|\leq0.05.
$$

If reproduction fails, P11 route must mark:

```text
baseline_step_drift_unresolved
```

and P2-P7 results are diagnostic only.

### 5.5 可视化

```text
p0_v616_v617_reproduction_bar.svg
p0_step_drift_waterfall.svg
p0_contract_heatmap.svg
p0_runner_kernel_matrix.svg
```

---

## 6. P1：GT4/E1 efficiency margin attribution

### 6.1 目的

P1 分解 GT4/E1 的 memory saving 和 step regression，判断 E1 是否值得继续组合。

### 6.2 测量对象

```text
MLP-manual-linear-reference
GT4-v616-reproduced
E0-GT4-v617-baseline
E1-GT4-recompute-hidden-cache
```

### 6.3 Component phases

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
phase_cleanup
```

### 6.4 必须记录字段

#### Efficiency

```text
memory_ratio
step_ratio
backward_ratio
forward_ratio
update_ratio
peak_allocated_MB
peak_reserved_MB
reserved_minus_allocated_MB
```

#### E1-specific

```text
hidden_cache_saved_MB
recompute_time_ms
recompute_kernel_count
recompute_torch_op_count
recompute_memory_temp_MB
memory_saved_per_extra_ms
backward_extra_time_ms_vs_E0
```

#### Memory source

```text
grouped_output_MB
grad_mix_MB
grad_poly_MB
update_temp_MB
optimizer_state_MB
triton_scratch_MB
manual_cache_MB
reserved_unallocated_MB
allocator_padding_MB
top1_memory_source
top2_memory_source
top3_memory_source
```

#### Runtime source

```text
forward_fused_grouped_ms
backward_fused_grouped_ms
update_prep_ms
optimizer_update_ms
kernel_launch_overhead_proxy_ms
layout_conversion_ms
contiguous_copy_ms
python_overhead_ms
```

### 6.5 判断标准

E1 continuation pass requires:

$$
\frac{M_{\text{E1}}}{M_{\text{E0}}}\leq0.985,
$$

and:

$$
\frac{T_{\text{step,E1}}}{T_{\text{step,E0}}}\leq1.02.
$$

If E1 fails this, it is:

```text
memory diagnostic only
```

and must not be included in P4 combo.

P1 attribution pass requires:

```text
top3_memory_sources identified
component timing explains >= 90% of step time
unknown_time_fraction <= 0.10
unknown_memory_fraction <= 0.10
```

### 6.6 可视化

```text
p1_e1_memory_vs_step_tradeoff.svg
p1_memory_saved_per_extra_ms.svg
p1_gt4_memory_gap_waterfall.svg
p1_gt4_step_time_waterfall.svg
p1_component_timing_stacked_bar.svg
```

---

## 7. P2：step-preserving memory repair

### 7.1 目的

P2 专门做 memory repair，但必须不牺牲 step。v6.17 的 E1 失败原因是 memory 降而 step 变差，所以 v6.18 的 memory repair 必须 step-preserving。

### 7.2 Candidate packages

```text
M0-GT4-baseline

M1-grad-mix-buffer-reuse:
  reuse grad_mix buffer without recomputing hidden

M2-grad-poly-buffer-reuse:
  reuse grad_poly buffer without recomputing hidden

M3-update-temp-reuse:
  reuse update temp and avoid separate update materialization

M4-triton-scratch-lifetime-trim:
  release / reuse Triton scratch earlier

M5-reserved-memory-trim:
  reduce reserved-unallocated gap

M6-short-lived-output-free:
  shorten grouped output lifetime after backward consumes it

M7-memory-combo-safe:
  M1 + M2 + M3 + M4 + M6

M8-E1-recompute-hidden-cache:
  include only as diagnostic, not default combo
```

### 7.3 必须记录字段

```text
package
implementation_status
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
memory_improvement_vs_GT4
step_improvement_vs_GT4
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

If a candidate improves memory but worsens step by more than $2\%$, it is S3 diagnostic only.

### 7.5 可视化

```text
p2_memory_repair_pareto.svg
p2_memory_source_reduction_bar.svg
p2_step_preservation_bar.svg
p2_s1_boundary_plot.svg
```

---

## 8. P3：memory-preserving step repair

### 8.1 目的

P3 专门降低 step ratio，但不能让 memory 回退。v6.16 GT4 已经接近 step gate，v6.17 E0 regression 需要被修复。

### 8.2 Candidate packages

```text
T0-GT4-baseline

T1-forward-fastpath-real:
  optimize forward grouped path, not just mark not_implemented

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
  explicitly ensure no hidden recompute path is active
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

S2/S1/S0 gates same as P2.

If step improves but memory exceeds $1.05$, it is runtime diagnostic only.

### 8.5 可视化

```text
p3_step_repair_pareto.svg
p3_step_time_waterfall.svg
p3_forward_update_launch_breakdown.svg
p3_memory_regression_bar.svg
```

---

## 9. P4：combo repair and survivor selection

### 9.1 目的

P4 组合 P2/P3 中通过 single-factor gate 的 repairs。v6.17 的问题是 best memory selection 选到了 S3，v6.18 必须按 survivor gate 选择。

### 9.2 Candidate packages

```text
C0-GT4-baseline
C1-best-step-preserving-memory-repair
C2-best-memory-preserving-step-repair
C3-safe-memory+step-combo
C4-S2-restore-combo
C5-S1-candidate-combo
C6-aggressive-S0-diagnostic
```

`C6` 只有在 C3/C4 稳定后才运行，不能作为默认主线。

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

Shape stability:

$$
S_{\text{shape,mem}}=1-\frac{\operatorname{std}(r_{\text{mem}})}{\operatorname{mean}(r_{\text{mem}})+\epsilon}.
$$

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

## 10. P5：one-step probe for S2/S1/S0

### 10.1 目的

恢复 S2 或得到 S1/S0 后，先验证 one-step descent。v6.17 没有跑 P7 because no S2；v6.18 若有 S2 必须恢复 one-step probe。

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

如果 P4/P5 恢复 S2，则运行 diagnostic task，继续 v6.16 的 task gap attribution。如果只得到 S3/S4，则不运行。

### 11.2 必跑方法

```text
MLP-autograd-reference
MLP-manual-linear-reference
GT4-v616-reference + ManualAdanLite
Best-v618-S2 + ManualAdanLite
Best-v618-S2 + ManualAdamW
```

如果 S1/S0 exists:

```text
Best-v618-S1 + ManualAdanLite
Best-v618-S1 + ManualAdamW
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

## 12. P7：lightweight expressivity / optimizer diagnostic under S2 envelope

### 12.1 目的

P7 只在 S2 恢复后运行。它不是 optimizer 大扫，而是判断 v6.16 task gap 是否有可修迹象。所有修复必须保持 S2 envelope。

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

`label-smoothing-diagnostic` 只能用于诊断 calibration/generalization，不能单独 claim final success，除非 MLP baseline 同样使用 label smoothing。

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

relative to v6.16 GT4+ManualAdanLite baseline or v6.18 best-S2 baseline.

If a repair improves accuracy but violates S2, it is diagnostic-only and cannot enter official route.

### 12.5 可视化

```text
p7_optimizer_expressivity_pareto.svg
p7_accuracy_vs_efficiency.svg
p7_feature_rank_improvement_bar.svg
p7_ece_train_val_gap_bar.svg
```

---

## 13. P8：official task re-entry gate

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
  Best-v618-S1 + ManualAdamW
  Best-v618-S1 + ManualAdanLite
  Best-v618-S1 + best diagnostic optimizer
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

P9 only opens after official P8 task pass. It is not v6.18 main stage.

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
  v6.18 restores S2 and P5 pass.
  Open diagnostic task.

R2-S1Solved:
  v6.18 reaches S1/S0 and P5 pass.
  Open official task.

R3-MemoryRepairStepRegression:
  memory repair improves memory but worsens step.
  Do not include in combo; repair is diagnostic.

R4-StepRepairMemoryRegression:
  step repair improves runtime but worsens memory.
  Do not include in combo; repair is diagnostic.

R5-BaselineStepDriftUnresolved:
  v6.16 GT4 S2 cannot be reproduced.
  Fix runner/protocol before task or optimizer.

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
  Focus expressivity/optimizer, not memory.
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
F5_recompute_step_regression
F6_memory_repair_step_regression
F7_step_repair_memory_regression
F8_efficiency_margin_fail
F9_task_gap_expressivity
F10_task_gap_optimization
F11_task_gap_generalization
F12_task_gap_unclear
F13_official_task_gated
F14_diagnostic_task_gated
F15_optimizer_gated
F16_functional_gated
F17_fake_or_proxy_violation
F18_artifact_missing
```

### Artifacts

```text
p0_contract.csv
p0_reproduction_check.csv
p0_runner_kernel_matrix.csv
p1_efficiency_margin_attribution.csv
p1_component_kernel_audit.csv
p2_step_preserving_memory_repair.csv
p2_step_preserving_memory_repair_detail.csv
p3_memory_preserving_step_repair.csv
p3_memory_preserving_step_repair_detail.csv
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
figures/p0_v616_v617_reproduction_bar.svg
figures/p0_step_drift_waterfall.svg
figures/p1_e1_memory_vs_step_tradeoff.svg
figures/p1_memory_saved_per_extra_ms.svg
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

### Case A：E1 memory 降但 step 继续差

如果 E1 或类似 recompute repair 仍满足：

$$
M_{\text{new}}<M_{\text{GT4}},
$$

但：

$$
T_{\text{step,new}}>T_{\text{step,GT4}},
$$

则该方向只保留为 diagnostic，不进入 combo。

### Case B：S2 恢复但 S1 未达成

如果恢复：

$$
r_{\text{mem}}\leq1.05,\quad r_{\text{step}}\leq1.50,
$$

但没有达到 S1，则允许 diagnostic task，不允许 official task。

### Case C：S1/S0 达成

如果：

$$
r_{\text{mem}}<1.00,\quad r_{\text{step}}\leq1.35,
$$

且 P5 pass，则首次打开 official task re-entry。

### Case D：S2 恢复且 diagnostic task 改善

如果 diagnostic task val acc 相对 v6.16 GT4 提升至少 $2\%$，fused grouped 继续主线，即使 S1 还未达成。

### Case E：S2 恢复但 task gap 仍大

如果 S2 恢复但 val/test gap 仍大于 $5\%$，下一轮主线应转向 expressivity/cross-group repair，而不是只修 memory。

### Case F：S2 无法恢复

如果 v6.18 无法复现或恢复 S2，则不能开 optimizer，也不能开 task。必须先解决 runner/protocol/kernel margin。

---

## 19. 最终建议

v6.18 的一句话策略是：

$$
\boxed{
\text{先恢复 v6.16 GT4 的 S2，再做 step-preserving memory repair 和 memory-preserving step repair。}
}
$$

v6.17 的 E1 证明 hidden-cache recompute 不是合适主修复：它降低了 memory，但伤害 step。下一轮不要继续沿这个方向做 combo。真正需要的是：

```text
grad/update buffer reuse
short-lived output lifetime
update temp reuse
forward/update fastpath
launch/call overhead reduction
```

只有 S2 恢复后，才重新打开 diagnostic task gap attribution；只有 S1/S0 达成后，才打开 official task 和系统 optimizer exploration。
