# DG-KAN v9.2.40 Parallel Base-Repair Confirmation 与 Snapshot Functional Re-entry 完整实验计划

> 本计划基于 v9.2.39 `Anchor-or-Pivot LQ Base Stability 与 Snapshot Functional Re-entry` 的真实复盘制定。  
> v9.2.39 的 terminal route 是：
>
> ```text
> route = R5-BaseRepairPass
> base_candidate = LQ-t2-h256
> success_v9239_strict_purekan_functional = False
> success_v9239_full_functional = False
> success_v9239_external_ready = False
> ```
>
> v9.2.39 的关键结果是：
>
> ```text
> Current LQ repeated anchor:
>   near-rate mean/min/max = 0.666667 / 0.666667 / 0.666667
>   macro delta mean = -0.004500
>   miss-row stability = 1.0
>
> Miss-row attribution:
>   MNIST seed2 = M1-KAN_acc_dropped
>   KMNIST seed0 = M4-threshold_borderline
>   KMNIST seed1 = M3-both_changed
>
> Protocol audit:
>   protocol mismatch detected = 1
>   unresolved = 0
>   MLP drift = 0
>
> Global base repair:
>   pass = 1
>   best = R2-LQ-fanin-output-scale-confirmed
>   near rows = 9/9
>   macro delta = -0.003056
>   step q90 = 1.012960
>
> Current blocker:
>   snapshot_late_attach_not_implemented_after_base_repair_pass
> ```
>
> 因此 v9.2.40 的目标不是继续在未锚定 base 上调 functional，也不是针对 MNIST / Fashion / KMNIST 分别调参，而是：
>
> $$
> \boxed{
> \text{并行确认 repaired base 的稳定性，并立即实现 snapshot late-attach functional re-entry。}
> }
> $$
>
> v9.2.40 必须改变推进方式。过去几轮每次只测一个 gate，逻辑干净但速度太慢。v9.2.40 应该采用 **parallel gated pipeline**：
>
> ```text
> Lane A:
>   R2 base repair robust confirmation.
>
> Lane B:
>   snapshot late-attach implementation and inactive/no-event equivalence.
>
> Lane C:
>   functional carrier actuatability scout.
>
> Lane D:
>   value observability and paired replay on eligible survivors.
>
> Lane E:
>   fallback base pivot if R2 fails robust confirmation.
> ```
>
> 所有 lane 可以并行跑，但 official route 仍由 gate 控制。也就是说：  
> diagnostic rows 可以提前测，但只有当前置 gate 通过后，才允许计入 official success。

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report miss-row attribution
  report leave-dataset-out generalization
  report signal-stratum distribution by dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a candidate because it rescues only one dataset
```

---

# Part I. 当前结果的独立判断

## 1. v9.2.39 没有达到 strict functional success

v9.2.39 的 `R5-BaseRepairPass` 只表示 base repair 成功，不表示 functional success。  
以下 downstream gates 仍然是 0：

```text
snapshot_attach_pass = 0
checkpoint_inactive_equivalence_pass = 0
no_event_replay_preservation_pass = 0
functional_carrier_pass = 0
value_observability_pass = 0
leave_dataset_out_pass = 0
paired_replay_pass = 0
short_run_pass = 0
full_run_pass = 0
external_ready = 0
```

因此不能声明：

```text
strict PureKAN functional success
full functional success
external-ready success
functional control pass
functional task-safe pass
functional system pass
```

当前唯一已经推进的是：

$$
\boxed{
\text{base anchor 从 broken 进入 repair survivor。}
}
$$

## 2. v9.2.39 的真实进展

v9.2.38 停在 `R1-LQBaseAnchorBroken`，因为 current LQ 只有 `6/9` near-pass。v9.2.39 做了三件重要事。

第一，miss-row autopsy 把 `6/9` 拆开了，而不是笼统说 LQ 崩了：

```text
MNIST seed2:
  current delta = -0.016000
  near margin = -0.006000
  mode = M1-KAN_acc_dropped

KMNIST seed0:
  current delta = -0.011500
  near margin = -0.001500
  mode = M4-threshold_borderline

KMNIST seed1:
  current delta = -0.012500
  near margin = -0.002500
  mode = M3-both_changed
```

这说明至少一个 miss row 是真正 KAN drop，而不是全部都是 near-threshold brittleness。

第二，P3 repeated re-anchor 证明 current LQ 的 `6/9` 不是单次偶然：

```text
reruns = 3
near_rate_mean = 0.666667
near_rate_min = 0.666667
miss_row_stability = 1.0
```

所以不能把 v9.2.38 的 anchor fail 当成纯噪声。

第三，P5 global base repair 找到真正 survivor：

```text
R2-LQ-fanin-output-scale-confirmed:
  P4 system = 1
  near rows = 9/9
  macro delta = -0.003056
  step q90 = 1.012960
  repair pass = 1
```

这一步非常关键。它说明 LQ/T2 路线不需要马上 pivot；但 functional route 必须基于 repaired base，而不是旧 current LQ。

## 3. 当前进展为什么显得慢

进展慢的本质不是因为实验没有信息，而是因为流程太串行：

```text
v9.2.35:
  value-aligned primitive 实现了，但破坏 P5。

v9.2.36:
  base-preserving subspace 实现了，但 route-level P5 失败。

v9.2.37:
  training-path equivalence 通过，但 P5 仍失败。

v9.2.38:
  LQ anchor broken，snapshot late attach 没打开。

v9.2.39:
  base repair pass，但 snapshot late attach 仍未实现。
```

这条链路很干净，但如果下一轮还是只实现 snapshot attach，不同时做 base robust confirmation、carrier scout、value scout，就会继续慢。

所以 v9.2.40 要改成并行路线：  
**base robust confirmation 与 snapshot attach 必须同轮跑；functional carrier scout 与 value scout 可以提前作为 diagnostic 跑，但不能越过 gate 写 success。**

## 4. 当前真正 blocker

当前 blocker 已经从：

```text
LQ base anchor broken
```

变为：

```text
snapshot_late_attach_not_implemented_after_base_repair_pass
```

但更深一层是：

$$
\boxed{
\text{我们还没有证明 repaired LQ base 能承载 functional re-entry。}
}
$$

这个证明包含四件事：

```text
1. R2 base repair 在更大 seed / repeated protocol 下仍稳定。
2. snapshot late attach inactive 时严格等价 repaired base checkpoint。
3. no-event replay 不改变 repaired base continuation。
4. event-time functional carrier 能产生 non-silent / non-AdamW movement，并且 value score 能击败 controls。
```

只有这四件事同时成立，functional 才真正重新进入主线。

## 5. 是否在正确道路上

是，但 v9.2.40 必须加速。  
正确道路是：

```text
base repair / anchor
→ snapshot late attach
→ carrier actuatability
→ value observability
→ leave-dataset-out
→ paired replay
```

错误道路是：

```text
Fashion-specific controller
KMNIST-specific primitive
MNIST abstain rule
继续调 horizon / threshold / target
```

v9.2.39 没有 dataset tuning，P5 base repair 也是 global repair，而不是 per-dataset 修补。这点是对的。下一步也必须保持。

## 6. 离目标还差多远

离 **local functional causality** 还差四道门：

```text
1. repaired base robust confirmation
2. snapshot late attach inactive/no-event equivalence
3. functional carrier actuatability
4. official paired replay beat AdamWParallel / best LR
```

v9.2.39 只完成了第 1 道门的 first-wave repair，还没有完成 robust confirmation。

离 **full functional success** 还差：

```text
short-run task-safe mechanism gain
full 10-seed macro / hard-stratum / geometry gain
robustness
strong baseline
external-ready
```

所以 v9.2.40 不能直接 full-run；但必须并行打开 late attach / carrier scout，不能继续只修 base。

---

# Part II. v9.2.40 总体目标

v9.2.40 的总体目标是：

$$
\boxed{
\text{并行确认 repaired LQ base，并完成 snapshot late-attach functional re-entry 的首轮闭环。}
}
$$

具体目标分为六层。

## 1. Repaired base robust confirmation

确认 `R2-LQ-fanin-output-scale-confirmed` 不是 3-seed 偶然。

成功标准：

$$
near\_pass\_rate_{\text{R2}}\geq0.80,
$$

$$
\Delta Acc_{\text{macro,R2}}\geq-0.01,
$$

并且：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

强确认标准：

$$
near\_pass\_rate_{\text{R2,10seed}}\geq0.80,
$$

$$
CI95_{\text{macro low}}\geq-0.01.
$$

## 2. Snapshot late-attach implementation

在 repaired base checkpoint 上 attach functional carrier。Attach 前 base phase 不存在 functional channel；attach 后 inactive 时必须等价 checkpoint：

$$
\max_x
\|z_{\text{attach-off}}-z_{\text{base checkpoint}}\|_\infty
\leq10^{-6}.
$$

任务参数与 optimizer state 必须不变：

$$
\|\theta_T^{after attach}-\theta_T^{before attach}\|_\infty=0,
$$

$$
\|m_T^{after attach}-m_T^{before attach}\|_\infty=0,
$$

$$
\|v_T^{after attach}-v_T^{before attach}\|_\infty=0.
$$

## 3. No-event replay preservation

没有 functional event 时，attach branch 必须和 base continuation 一致：

$$
\max_{t\leq T}
\|z_{t,\text{attach-no-event}}-z_{t,\text{base-continue}}\|_\infty
\leq10^{-5}.
$$

Metric preservation：

$$
|CEp99_{\text{attach-no-event}}-CEp99_{\text{base-continue}}|
\leq10^{-5},
$$

$$
|MarginP10_{\text{attach-no-event}}-MarginP10_{\text{base-continue}}|
\leq10^{-5}.
$$

## 4. Functional carrier actuatability

Event-time functional carrier 必须 non-silent：

$$
r_{z,\text{tail}}\geq0.10.
$$

并且不是 AdamWParallel 伪装：

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

## 5. Value observability

Functional score 必须预测 grounded Real-vs-control value：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept / abstain gate：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

## 6. Official paired replay

只有 P1-P7 pass 后，paired replay 才能 official 打开。  
成功标准：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

---

# Part III. 核心假设

## H1：v9.2.39 的 base repair 是真实进展，但需要 robust confirmation

P5 中 `R2-LQ-fanin-output-scale-confirmed` 达到 `9/9` near rows、macro delta `-0.003056`、step q90 `1.012960`。这明显优于 current LQ repeated anchor 的 `6/9`。  
H1 认为 R2 可以成为新的 functional base anchor。

H1 成立标准：

$$
near\_pass\_rate_{\text{R2,repeat}}\geq0.80,
$$

并且 10-seed 或 repeated protocol 下不低于：

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

如果 R2 在 repeated protocol 下跌回 `6/9` 或 worse，则 H1 失败，不能进入 official functional route。

## H2：snapshot late attach 可以避免 base contamination

过去 VAOP/BPFS/TPEA 的失败都与 base phase / P5 gate 纠缠。  
H2 认为 checkpoint-level late attach 是最干净路线，因为 functional channel 不参与 base phase。

H2 成立标准：

$$
\max_x
\|z_{\text{attach-off}}-z_{\text{base checkpoint}}\|_\infty
\leq10^{-6},
$$

并且 no-event replay preservation pass。

## H3：functional carrier 的 first useful target 不是 accuracy，而是 non-silent non-AdamW movement

如果 carrier 不能产生 output movement，value score 再好也没意义。  
H3 成立标准：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

## H4：control-gap / role-wise score 应优先于 movement-only score

v9.2.35 已经显示 control-gap correlation 比 orthogonal-tail movement 更有信息。  
H4 成立标准：

$$
AUC(S_{\text{control-gap}})
-
AUC(r_{\perp,\text{tail}})
\geq0.10
$$

or:

$$
Corr(S_{\text{control-gap}},V)
-
Corr(r_{\perp,\text{tail}},V)
\geq0.10.
$$

## H5：所有 functional controller 必须 dataset-agnostic

即使 miss rows 或 functional gains 主要来自 KMNIST，也不能写 dataset branch。  
H5 成立标准：

```text
dataset_name_used = 0
posthoc_metric_used_at_commit = 0
validation_used = 0
test_used = 0
leave_dataset_out_pass = 1 before official success
```

## H6：如果 repaired base pass 但 carrier/value fail，路线仍然正确

如果 R2 robust base pass、late attach pass，但 carrier silent 或 value unobservable，这说明我们终于把 blocker 从 base 层推进到 functional mechanism 层。  
这种失败比当前状态更有价值，因为它给出合法 functional diagnosis。

---

# Part IV. 并行执行设计

v9.2.40 必须从串行改为并行。  
同一个 runner 应该同时运行以下 lane：

```text
Lane A:
  Repaired base robust confirmation.

Lane B:
  Snapshot late attach implementation and inactive/no-event equivalence.

Lane C:
  Functional carrier actuatability scout.

Lane D:
  Value observability scout.

Lane E:
  Official paired replay on eligible survivor.

Lane F:
  Fallback base repair / pivot if R2 fails.
```

注意：

```text
Lane C / D 可以提前 diagnostic 跑；
但只有 Lane A / B pass 后，C / D 才能 official 计入 route。
```

这可以加快实验，同时不破坏 gate discipline。

---

# Part V. Candidate 设计

## 1. Base candidates

### B0：Current LQ reference

```text
R0-LQ-current
```

用于对照，不作为新 functional base。

### B1：Best repaired base

```text
R2-LQ-fanin-output-scale-confirmed
```

v9.2.39 best repair。v9.2.40 的 primary base。

### B2：Backup repaired base

```text
R5-LQ-hidden256
```

v9.2.39 中 P4 pass、8/9 near rows，可作为 backup。

### B3：Conservative repair reference

```text
R0-LQ-current repeated with confirmed protocol
```

用于判断 R2 gain 是否真实来自 repair，而不是 protocol variance。

---

## 2. Attach candidates

### A0：No attach base checkpoint

Reference：

$$
z(x)=z_{\text{base}}(x;\theta_T^\star).
$$

### A1：LateAttachZeroLinearTail

$$
\phi_{ij}(h)
=
\phi^{base}_{ij}(h;\theta_T^\star)
+
\lambda(e)a_{ij}q(h)h.
$$

Inactive：

$$
\lambda(e)=0.
$$

### A2：LateAttachControlGapChannel

Accept condition：

$$
LCB(Gain_{\text{Real}})
>
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### A3：LateAttachRoleWiseFT7EdgeCarrier

$$
\phi_{ij}(h)
=
\phi^{base}_{ij}(h;\theta_T^\star)
+
\lambda(e)
[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
].
$$

Role weights use signal features only:

$$
\alpha_r(e)
=
f(
CEp99,
MarginP10,
Curvature,
BranchRatio,
EffectiveDerivative,
ControlGap
).
$$

### A4：SymbolicFunctionalSpecAttach

Functional channel exists as a spec until event trigger:

```text
base phase:
  no theta_F

event phase:
  instantiate theta_F as edge-owned zero-init
  apply one functional update
```

### A5：FamilyValueLateAttach

Family:

$$
family(e)=(stratum(e), horizon(e), risk\_bucket(e), role\_bucket(e)).
$$

Accept if:

$$
\mathbb{E}[V_{\text{grounded}}\mid family]>0
$$

and:

$$
Reliability(family)\geq0.30.
$$

---

# Part VI. 实验阶段

## P0：v9.2.39 boundary reproduction

### 目标

复现 v9.2.39 boundary，确认当前不是读取错误。

### 必须记录

```text
route
source_route_v9238
current_lq_near_rate
current_lq_macro_delta
miss_row_count
miss_row_borderline_rate
protocol_mismatch_detected
protocol_mismatch_unresolved
current_lq_rerun_count
best_base_repair
best_base_near_rows
best_base_macro_delta
best_base_step_q90
base_repair_pass
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R5-BaseRepairPass
best_base_repair = R2-LQ-fanin-output-scale-confirmed
base_repair_pass = 1
snapshot_attach_pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_anchor_to_repair_gate_ladder.svg
p0_base_repair_scorecard.svg
```

---

## P1：Parallel base robust confirmation

### 目标

并行确认 R2 base repair 是否稳定。

### 设置

```text
base_candidates:
  R0-LQ-current
  R2-LQ-fanin-output-scale-confirmed
  R5-LQ-hidden256

datasets:
  MNIST,Fashion-MNIST,KMNIST

seeds:
  0..9 for primary confirmation

reruns:
  3 if compute allows, otherwise 1 full 10-seed + 3-seed repeat
```

### 必须记录

```text
candidate
dataset
seed
rerun_id
acc
mlp_acc
delta_vs_mlp
near_pass
near_margin
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
P4_forward_q90
P4_backward_q90
P4_step_q90
memory_ratio
```

Aggregate：

```text
near_pass_rate
macro_delta_mean
macro_delta_std
CI95_macro_low
miss_row_count
miss_row_stability
step_q90
memory_ratio
```

### 判断标准

Base robust pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
CI95_{\text{macro low}}\geq-0.01,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Promotion rule：

```text
Promote candidate with best global macro / system / conditioning.
No dataset-specific promotion.
No single-dataset rescue promotion.
```

### 可视化

```text
p1_base_robust_nearpass_matrix.svg
p1_macro_delta_ci.svg
p1_miss_row_stability.svg
p1_system_pareto.svg
```

---

## P2：Snapshot late attach implementation

### 目标

实现 A1-A5，至少 A1/A2/A3 必须实现。  
Base phase 使用 P1 survivor checkpoint。

### 必须记录

```text
attach_candidate
base_candidate
base_checkpoint_hash
task_param_hash_before_attach
task_param_hash_after_attach
optimizer_state_hash_before_attach
optimizer_state_hash_after_attach
functional_param_registered_before_attach
functional_param_registered_after_attach
functional_param_zero_init
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
lambda_inactive_exact_zero
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
A1 implemented = 1
A2 or A3 implemented = 1
strict PureKAN contract pass = 1
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward/backward/update = 1
uses_loss_backward = 0
```

### 可视化

```text
p2_attach_implementation_matrix.svg
p2_state_hash_stability.svg
```

---

## P3：Checkpoint inactive equivalence

### 目标

证明 attach 后 $\lambda=0$ 不改变 base checkpoint。

### 必须记录

```text
attach_candidate
base_candidate
dataset
seed
batch_id
base_checkpoint_hash
max_logit_diff_inactive
mean_logit_diff_inactive
task_param_max_diff
optimizer_m_max_diff
optimizer_v_max_diff
functional_zero_norm
functional_grad_zero_norm
lambda_value
lambda_leak_detected
```

### 判断标准

Inactive equivalence pass：

$$
\max_x
\|z_{\text{attach-off}}-z_{\text{base checkpoint}}\|_\infty
\leq10^{-6}.
$$

Task state stability：

$$
\|\theta_T^{after attach}-\theta_T^{before attach}\|_\infty=0.
$$

Optimizer state stability：

$$
\|m_T^{after attach}-m_T^{before attach}\|_\infty=0,
$$

$$
\|v_T^{after attach}-v_T^{before attach}\|_\infty=0.
$$

### 可视化

```text
p3_inactive_equivalence_logit_diff.svg
p3_task_param_state_diff.svg
p3_lambda_leak_audit.svg
```

---

## P4：No-event replay preservation

### 目标

证明 attach branch 在 no-event setting 下不改变 base continuation。

### 设置

```text
branches:
  base_continue
  attach_off_continue
  attach_no_event_continue

steps:
  1,5,20,80

functional_update:
  off

events:
  none
```

### 必须记录

```text
attach_candidate
base_candidate
dataset
seed
steps
branch
train_loss
holdout_loss
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
logit_diff_vs_base_continue
task_param_diff_vs_base_continue
optimizer_state_diff_vs_base_continue
step_q90
memory_ratio
```

### 判断标准

No-event preservation pass：

$$
\max_{t\leq T}
\|z_{t,\text{attach-no-event}}-z_{t,\text{base-continue}}\|_\infty
\leq10^{-5}.
$$

Metric preservation：

$$
|CEp99_{\text{attach-no-event}}-CEp99_{\text{base-continue}}|
\leq10^{-5}.
$$

$$
|MarginP10_{\text{attach-no-event}}-MarginP10_{\text{base-continue}}|
\leq10^{-5}.
$$

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_no_event_replay_equivalence.svg
p4_metric_preservation_trace.svg
p4_attach_system_overhead.svg
```

---

## P5：Functional carrier actuatability scout

### 目标

在 base-ready attach 上并行测试 functional carrier 是否能动。  
P5 可以 diagnostic 先跑，但 official 计入必须要求 P1-P4 pass。

### 设置

```text
base = P1 survivor checkpoint
attach_candidates = P4 survivors
events = signal strata S1-S8
horizons = 1,5,20,80,240
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
```

### 必须记录

```text
attach_candidate
event_id
signal_stratum
horizon
branch
actual_logit_delta_norm
actual_tail_logit_delta_norm
actual_nonadamw_delta_norm
r_z
r_z_tail
r_perp_tail
cos_real_adamw
cos_real_bestlr
branch_ratio
effective_derivative
functional_step_norm
task_step_norm
task_safe
bad_event
official_eligible
gate_missing_reason
```

### 判断标准

Carrier non-silent：

$$
r_{z,\text{tail}}\geq0.10.
$$

Non-AdamW component：

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p5_functional_movement_distribution.svg
p5_tail_movement_vs_task_safety.svg
p5_nonadamw_component.svg
p5_carrier_by_attach_candidate.svg
```

---

## P6：Value observability scout

### 目标

判断 carrier score 是否预测 grounded Real-vs-control value。  
同样可以 diagnostic 并行跑，但 official 计入需要 P1-P5 pass。

### 必须记录

```text
attach_candidate
controller
event_id
signal_stratum
horizon
value_score
control_gap_score
tail_value_score
role_score
grounded_value
Y_beat
corr
auc
precision
coverage
bad_event_rate
accepted_event_count
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
step_q90
memory_ratio
official_eligible
gate_missing_reason
```

### 判断标准

Observability pass：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept / abstain pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Legality：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

### 可视化

```text
p6_value_score_vs_grounded_value.svg
p6_auc_precision_coverage.svg
p6_score_components_correlation.svg
p6_accepted_strata_distribution.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 controller 隐式 dataset tuning。

### 设置

Leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
attach_candidate
controller
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
shuffle_control_pass
dataset_name_used
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two LDO splits satisfy：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and CE-tail non-worse：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p7_leave_dataset_out_matrix.svg
p7_leave_stratum_out_matrix.svg
p7_hidden_dataset_tuning_audit.svg
```

---

## P8：Official snapshot late-attach paired replay

### 目标

只有 P1-P7 survivor 可以进入 official paired replay。

### 设置

```text
base = P1 repaired base checkpoint
attach_candidate = best P7 survivor
controller = best legal signal-based controller
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled
```

### 必须记录

```text
attach_candidate
controller
dataset
seed
horizon
signal_stratum
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Official paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p8_official_paired_replay_pareto.svg
p8_macro_beat_rate.svg
p8_signal_stratum_win_matrix.svg
p8_shuffle_control_matrix.svg
p8_system_gate_distribution.svg
```

---

## P9：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P8 survivor 进入。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, ValueScoreShuffled
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass，至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p9_short_run_task_mechanism_pareto.svg
p9_short_run_controls.svg
p9_event_timeline.svg
p9_ce_tail_margin_panel.svg
```

---

## P10：Full 10-seed validation

### 目标

验证 strict PureKAN functional route 是否在 full training 上成立。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass，至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

## P11：Robustness / strong baseline / external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 必须记录

```text
candidate
baseline_id
params
flops
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
clean_acc
noisy_acc
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
local_lipschitz
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9240.csv
p0_v9239_boundary_reproduction.csv
p1_parallel_base_robust_confirmation.csv
p2_snapshot_late_attach_implementation.csv
p3_checkpoint_inactive_equivalence.csv
p4_no_event_replay_preservation.csv
p5_functional_carrier_actuatability.csv
p6_value_observability_audit.csv
p7_leave_dataset_and_stratum_out_validation.csv
p8_official_snapshot_late_attach_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
p11_robustness_external_ready.csv
base_repair_confirmation_trace_v9240.csv
snapshot_attach_trace_v9240.csv
checkpoint_equivalence_trace_v9240.csv
functional_carrier_trace_v9240.csv
value_score_trace_v9240.csv
leave_dataset_out_trace_v9240.csv
paired_replay_branch_trace_v9240.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9239_boundary_unstable
F3_dataset_tuning_detected
F4_repaired_base_not_robust
F5_base_repair_system_fail
F6_snapshot_late_attach_not_implemented
F7_checkpoint_inactive_equivalence_fail
F8_no_event_replay_preservation_fail
F9_functional_carrier_silent
F10_value_observability_fail
F11_value_score_system_too_expensive
F12_leave_dataset_out_fail
F13_leave_stratum_out_fail
F14_paired_replay_control_equivalent
F15_shuffle_control_pass
F16_functional_lr_equivalent
F17_short_run_task_drop
F18_full_run_no_macro_hard_stratum_gain
F19_strong_baseline_explains_gain
F20_robustness_fail
F21_external_not_ready
F22_fake_or_proxy_violation
F23_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-RepairedBaseRobustConfirmed:
  R2 or another global repair passes robust base confirmation.

R2-RepairedBaseNotRobust:
  v9.2.39 repair fails repeated / 10-seed confirmation.

R3-SnapshotLateAttachImplemented:
  late attach implemented under strict PureKAN contract.

R4-SnapshotAttachInactiveEquivalent:
  attach-off equals repaired base checkpoint.

R5-NoEventReplayPreserved:
  no-event replay matches base continuation.

R6-FunctionalCarrierActive:
  functional carrier produces non-silent, non-AdamW movement.

R7-ValueObservabilityPass:
  value score predicts grounded Real-vs-control value.

R8-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R9-SnapshotLateAttachPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-BasePreservedButFunctionalSilent:
  base preserved, but carrier cannot move output.

R11-BasePreservedButValueUnobservable:
  carrier moves output, but value score fails.

R12-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R13-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R14-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9239_boundary_pass
dataset_tuning_detected
best_repaired_base
repaired_base_robust_pass
repaired_base_near_rate
repaired_base_macro_delta
repaired_base_step_q90
snapshot_attach_pass
checkpoint_inactive_equivalence_pass
no_event_replay_preservation_pass
functional_carrier_pass
max_r_z_tail
max_r_perp_tail
value_observability_pass
value_auc
value_corr
accepted_precision
accepted_coverage
bad_event_rate
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9240_strict_purekan_functional
success_v9240_full_functional
success_v9240_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 base robust confirmation
  P2 snapshot attach implementation
  P3 checkpoint inactive equivalence smoke

Batch 2:
  P4 no-event replay preservation
  P5 functional carrier scout
  P6 value observability scout

Batch 3:
  P7 leave-dataset-out / leave-stratum-out
  P8 official paired replay

Batch 4:
  P9 short-run
  P10 full 10-seed
  P11 robustness / strong baseline
```

Gate rule：

```text
P5/P6 diagnostic rows may be measured before P1/P4 finish,
but official_eligible = 0 until P1-P4 pass.
```

This avoids slow serial iteration while preserving causal discipline.

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.39 boundary reproduced
R2 repaired base robust confirmation completed
snapshot late attach implemented
inactive equivalence measured
no fake/proxy/offload/loss/teacher violation
```

## Base-ready attach success

```text
repaired base robust pass
+
snapshot late attach inactive equivalence pass
+
no-event replay preservation pass
```

## Functional carrier success

```text
Base-ready attach success
+
functional carrier non-silent
+
task safety holds
```

## Local functional success

```text
Functional carrier success
+
value observability pass
+
leave-dataset-out pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.39 boundary cannot be reproduced；
2. R2 repaired base fails robust confirmation；
3. all global repaired bases fail P4/P5；
4. snapshot late attach cannot be implemented as strict PureKAN；
5. checkpoint inactive equivalence fails；
6. no-event replay preservation fails；
7. base preserved but functional carrier remains silent；
8. carrier moves but value score remains unobservable；
9. leave-dataset-out fails；
10. paired replay remains control-equivalent；
11. shuffle controls pass, indicating overfit；
12. full run gives no macro / hard-stratum / geometry gain；
13. functional breaks system gate；
14. gains are explained by QuadraticFeatureMLP；
15. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：R2 robust base passes and late attach passes

可以声明：

```text
v9.2.39 successfully repaired the base, and v9.2.40 restored a legal functional re-entry path.
```

但不能声明 functional success，除非 carrier / value / paired replay 也过。

## Case B：R2 robust base fails

必须声明：

```text
v9.2.39 base repair was first-wave only and not robust enough.
```

下一步转 global base repair / pivot，不继续 functional。

## Case C：late attach fails inactive equivalence

必须声明：

```text
functional attach infrastructure still contaminates repaired base.
```

下一步修 attach，不测 carrier。

## Case D：base and attach pass but carrier silent

必须声明：

```text
base is ready, but functional carrier lacks event-time actuatability.
```

下一步修 carrier，不做 dataset patch。

## Case E：carrier active but value unobservable

必须声明：

```text
functional movement exists but is not control-resistant value.
```

下一步回到 role-wise / control-gap mechanism。

## Case F：paired replay passes

可以声明：

```text
strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XII. 最终建议

v9.2.40 的一句话策略是：

$$
\boxed{
\text{把 v9.2.39 的 base repair 变成 robust anchor，同时并行实现 snapshot functional re-entry。}
}
$$

当前最关键的问题不是：

```text
Fashion 怎么调过；
KMNIST 怎么调过；
MNIST 是否 abstain；
再换哪个 target；
再调哪个 horizon；
再调哪个 value score。
```

而是：

```text
1. R2-LQ-fanin-output-scale-confirmed 是否在 10-seed / repeated protocol 下仍稳定？
2. snapshot late attach 是否能严格保持 repaired base checkpoint？
3. no-event replay 是否完全等价 base continuation？
4. functional carrier 是否能产生 non-silent, non-AdamW movement？
5. control-gap / role-wise score 是否能预测 grounded Real-vs-control value？
6. 这个规则能否 leave-dataset-out 泛化？
7. RealFunctional 能否在 official paired replay 中超过 AdamWParallel / best LR？
```

如果 v9.2.40 仍然只停在 base 或 attach gate，就说明我们应该暂停 functional re-entry，集中做 base / attach infrastructure。  
如果 v9.2.40 打开 carrier 但 value 失败，才真正回到 functional mechanism 本身。  
如果 v9.2.40 paired replay 通过，那就是 v9.2 strict PureKAN functional 主线的第一个真正局部因果成功。
