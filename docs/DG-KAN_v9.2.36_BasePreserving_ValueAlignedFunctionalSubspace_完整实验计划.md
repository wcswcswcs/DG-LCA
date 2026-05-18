# DG-KAN v9.2.36 Base-Preserving Value-Aligned Functional Subspace：从 VAOP 破坏 base 到可验证 functional carrier 的完整实验计划

> 本计划基于 v9.2.35 `Value-Aligned Observable Primitive` 的真实复盘制定。  
> v9.2.35 的 terminal route 是：
>
> ```text
> route = R9-VAOPAllFailPrimitiveReset
> base_candidate = LQ-t2-h256
> success_v9235_strict_purekan_functional = False
> success_v9235_full_functional = False
> success_v9235_external_ready = False
> ```
>
> v9.2.35 的核心事实是：**OP failure attribution 已经指向 control-gap unmodeled，VAOP1-VAOP6 也已经实现并通过 contract/grad audit；但 VAOP family 没有保住 P5 near-pass，best measured VAOP6 只过 P4，不过 P5，因此没有 base-qualified VAOP 进入 value observability / paired replay。**
>
> 因此，v9.2.36 不能继续围绕 Fashion / KMNIST / MNIST 做数据集特化调参，也不能继续简单堆一个更复杂的 value score。当前真正要解决的问题是：
>
> $$
> \boxed{
> \text{如何让 functional primitive 承载 value-aligned update，而不破坏 LQ/T2 base trainability？}
> }
> $$
>
> 本轮的核心路线是：保留 `LQ-t2-h256` 作为 task base scaffold，重做 **base-preserving functional subspace**。也就是说，functional channel 必须在 inactive / no-event / AdamW-only phase 下严格等价于 LQ/T2 base；只有当 signal-value gate 触发时，才通过 edge-owned functional subspace 做 functional update。  
>
> 这不是 residual MLP，不是 external sidecar，也不是 loss modification。它仍然是 PureKAN edge function：
>
> $$
> \phi_{ij}(h)
> =
> \phi^{task}_{ij}(h;\theta_T)
> +
> \lambda(e)\phi^{func}_{ij}(h;\theta_F),
> $$
>
> 其中 $\lambda(e)=0$ 时模型必须精确退化为 LQ/T2 base；$\lambda(e)>0$ 只由 dataset-agnostic signal event 决定。

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
  report leave-dataset-out generalization
  report which signal strata dominate failure

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
```

---

## 1. v9.2.35 独立判断

### 1.1 v9.2.35 没有达到目标

v9.2.35 的真实结果是：

```text
P0:
  v9.2.34 boundary reproduced
  source route = R6-ObservablePrimitiveEffectUnpredictable

P1:
  OP failure attribution pass = 1
  primary mode = F4-control_gap_unmodeled
  corr_r_perp_tail = -0.031948
  corr_obs_score = 0.000848
  corr_control_gap = 0.288810

P2:
  VAOP implemented count = 6

P3:
  contract / grad pass = 1 / 1

P4:
  best measured VAOP = VAOP6-FamilyValueChannel
  P4 pass = 1
  P5 near-pass = 0
  base-qualified VAOP = none

P5:
  value observability pass = 0
  best controller = none
  corr = 0
  AUC = 0.5

route:
  R9-VAOPAllFailPrimitiveReset
  blocker = all_vaop_failed_P5_nearpass
```

这说明 v9.2.35 没有失败在 “VAOP 没实现” 或 “contract/grad 不干净”。失败发生在更关键的位置：**value-aligned primitive 一旦作为 base 结构加入，就破坏了 AdamW-only base qualification。**

### 1.2 当前基函数没有整体放弃

`LQ-t2-h256` 仍然是当前最可靠的 strict FC-PureKAN base scaffold。当前要放弃的是：

```text
把 value-aligned functional primitive 直接混入 base task channel 训练
```

而不是放弃：

```text
LQ/T2 base
PureKAN edge-function structure
functional update core
```

正确表述是：

$$
\boxed{
\text{保留 LQ/T2 task base；重做 functional carrier，使其 base-preserving。}
}
$$

### 1.3 当前真正 blocker

v9.2.34 的 blocker 是：

```text
movement-observable primitive 不预测 value
```

v9.2.35 进一步说明：

```text
value-aligned primitive 尝试后，base trainability 先坏掉
```

因此当前 blocker 不是单纯 observability，而是：

$$
\boxed{
\text{value alignment 与 base preservation 没有同时成立。}
}
$$

VAOP 试图解决 control-gap unmodeled，但把 functional value channel 直接作为模型结构的一部分参与 base qualification，导致 P5 near-pass 全灭。下一步必须先让 functional channel 在 inactive 时不影响 base，再测试它是否能在 event-time 产生 value-aligned causal update。

---

## 2. v9.2.36 总体目标

v9.2.36 的总体目标是：

$$
\boxed{
\text{构造 base-preserving value-aligned functional subspace，并验证它能在不使用 dataset name 的条件下产生 control-resistant functional causality。}
}
$$

这个目标分五层。

### 2.1 Base preservation success

functional subspace inactive 时，模型必须等价于 LQ/T2 base：

$$
\max_x \|z_{\text{BPFS-off}}(x)-z_{\text{LQ}}(x)\|_\infty \leq 10^{-6}.
$$

AdamW-only training 也必须不被破坏：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

同时相对原 LQ base 的差异不能超过：

$$
|\Delta Acc_{\text{BPFS-off}}-\Delta Acc_{\text{LQ}}|\leq0.002.
$$

### 2.2 Functional carrier success

functional subspace active 时必须能产生可测 value movement：

$$
r_{z,\text{tail}}\geq0.10
$$

且不能只复制 AdamWParallel：

$$
r_{\perp,\text{tail}}\geq0.10.
$$

但 movement 不是成功标准；它只是 functional carrier 不静音的最低门槛。

### 2.3 Value observability success

pre-commit score 必须预测 grounded Real-vs-control value：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

或：

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept/abstain gate：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 2.4 Dataset-agnostic success

official controller 不能用 dataset name。必须通过 leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out splits 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

### 2.5 Local functional causality success

official paired replay 成功标准：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

并且：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

只有通过 paired replay 后，才允许进入 short-run / full-run。

---

## 3. 核心假设

### H1：v9.2.35 失败主要是 functional channel 污染 base，而不是 value alignment 概念无效

VAOP1-VAOP6 已经实现并通过 contract/grad，但没有 base-qualified survivor。H1 认为 VAOP 把 functional value channel 直接合入 base task channel，导致 AdamW-only P5 near-pass 被破坏。

H1 成立标准：

如果 BPFS inactive phase 满足：

$$
\max_x \|z_{\text{BPFS-off}}-z_{\text{LQ}}\|_\infty\leq10^{-6}
$$

并且 P5 near-pass 恢复到 LQ baseline 附近，则说明 v9.2.35 的主要问题是 base contamination。

### H2：functional subspace 应该 event-time 激活，而不是 base-phase 常驻训练

functional channel 不应在 base AdamW phase 中自由学习，否则它会破坏 LQ/T2 已经建立的 conditioning / trainability。  
H2 的机制是：

```text
task channel:
  AdamW-equivalent update, always active

functional channel:
  zero-init or dormant during base phase
  only updated by functional update rule
  only active under value-aligned signal event
```

H2 成立标准：

BPFS candidate 在 functional off 时过 P4/P5；在 event-time active 时产生 value movement，并且不伤 task safety。

### H3：control gap 必须被 functional update 直接建模，而不是间接靠 tail / orthogonal movement

v9.2.35 的 P1 说明 control-gap correlation 是最有信息的量：`corr_control_gap = 0.288810`，而 tail orthogonal corr 和 obs score corr 接近 0。  
H3 认为 functional score 应直接估计：

$$
S_{\text{gap}}
=
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}}, Gain_{\text{bestLR}}).
$$

H3 成立标准：

BPFS score 的 AUC / corr 显著高于 movement-only score：

$$
AUC(S_{\text{gap}})-AUC(r_{\perp,\text{tail}})\geq0.10
$$

或：

$$
Corr(S_{\text{gap}},V)-Corr(r_{\perp,\text{tail}},V)\geq0.10.
$$

### H4：v8-FT7 role-wise mechanism 需要以 base-preserving edge-owned 形式重引入

v8-FT7 曾经保留 functional core，但它不是当前 strict FC-PureKAN official route。  
H4 认为 FT7 的关键不是 “某个 target”，而是：

```text
role-wise functional update
branch / derivative / curvature gating
task channel 与 functional channel 分工
```

H4 成立标准：

RoleWise-BPFS 在不破坏 LQ base 的情况下，通过 observability / paired replay gate。

### H5：如果 base-preserving subspace 也不能产生 value-aligned gain，应回到更深 edge-function interface reset

如果 BPFS 既保住 base，又仍然无法预测 / 产生 Real-vs-control value，则不能继续 dataset patch。下一步应重新设计 edge-function interface 或重新抽取 v8-FT7 mechanism，而不是在 Fashion / KMNIST 上调参。

---

## 4. Candidate 设计

## 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  current strict FC-PureKAN AdamW-only base。

OP4-current-best:
  v9.2.34 best observable OP reference。

VAOP6-current-best-P4:
  v9.2.35 best measured VAOP, P4 pass but P5 fail。

V8-FT7:
  mechanism source only, not official strict candidate。
```

## 4.2 Base-preserving functional subspace candidates

### BPFS0：LQ Base Only

Reference:

$$
\phi_{ij}(h)=\phi^{LQ}_{ij}(h).
$$

No functional channel.

### BPFS1：Zero-Init Dormant TailLinear Functional Channel

Edge function:

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T)
+
\lambda(e)a_{ij}q(h)h.
$$

Initialization:

$$
a_{ij}=0.
$$

Inactive:

$$
\lambda(e)=0.
$$

Therefore:

$$
\phi_{ij}(h)=\phi^{LQ}_{ij}(h)
$$

exactly.

### BPFS2：Frozen-Task Functional Attach

Train LQ task channel first. Attach functional channel after base warmup:

```text
phase A:
  train LQ task channel only

phase B:
  freeze or low-lr anchor task channel
  allow functional channel event-time update
```

No dataset-specific route. Event comes from signal strata.

### BPFS3：Task-Orthogonal Functional Subspace

Define functional update projection:

$$
\Delta\theta_F^\perp
=
\Delta\theta_F
-
\operatorname{proj}_{\Delta\theta_T}
\Delta\theta_F.
$$

Only use tail-relevant component:

$$
\Delta\theta_F^{tail}=P_{tail}\Delta\theta_F^\perp.
$$

### BPFS4：Control-Gap Functional Channel

Functional channel only accepts if lower-bound value beats controls:

$$
LCB(Gain_{\text{Real}})>
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### BPFS5：Family-Value Functional Channel

Event family:

$$
family(e)=(stratum(e), horizon(e), risk\_bucket(e), primitive\_bucket(e)).
$$

Accept if:

$$
\mathbb{E}[V_{\text{grounded}}\mid family]>0
$$

and reliability:

$$
Reliability(family)\geq0.30.
$$

No dataset name allowed.

### BPFS6：RoleWise-FT7 Edge-Owned Functional Channel

Edge function:

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T)
+
\lambda(e)
\left[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
\right].
$$

Role weights:

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

No dataset name.

---

## 5. 实验阶段

## P0：v9.2.35 boundary reproduction

### 目标

复现 latest boundary，确认 v9.2.35 失败不是 measurement artifact。

### 必须记录

```text
route
source_route_v9234
op_failure_mode
corr_r_perp_tail
corr_obs_score
corr_control_gap
vaop_implemented_count
vaop_contract_pass
vaop_grad_pass
best_p4_vaop_candidate
best_p4_vaop_p4_pass
best_p4_vaop_p5_nearpass
base_qualified_vaop
value_observability_pass
best_controller
AUC
corr
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass:

```text
route = R9-VAOPAllFailPrimitiveReset
op_failure_mode = F4-control_gap_unmodeled
base-qualified VAOP = none
primary_blocker = all_vaop_failed_P5_nearpass
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_vaop_gate_ladder.svg
p0_control_gap_vs_movement_corr.svg
```

---

## P1：VAOP failure autopsy：base contamination vs value failure

### 目标

判断 VAOP 为何 P5 near-pass 全失败。必须区分：

```text
base channel 被污染
functional channel initialization 太强
basis conditioning 变坏
task / functional channel collision
value score 本身错误
```

### 必须记录

```text
primitive
dataset
seed
P4_pass
P5_nearpass
delta_vs_LQ_base
CEp99
margin_p10
ECE
NLL
basis_entropy
functional_channel_entropy
dominant_basis_fraction
lift_condition_number
effective_rank
task_channel_norm
functional_channel_norm
task_func_cosine
base_output_diff_at_init
base_output_diff_after_adamw
grad_norm_task
grad_norm_func
```

### 判断标准

Base contamination confirmed if:

$$
\max_x \|z_{\text{VAOP}}-z_{\text{LQ}}\|_\infty>10^{-4}
$$

before functional event, or:

$$
|\Delta Acc_{\text{VAOP-off}}-\Delta Acc_{\text{LQ}}|>0.002.
$$

Channel collision confirmed if:

$$
|\cos(g_T,g_F)|>0.50
$$

and P5 near-pass fails.

Conditioning failure confirmed if:

$$
\kappa_{\text{lift}}^{VAOP}>
2\kappa_{\text{lift}}^{LQ}
$$

or functional channel entropy collapses:

$$
H_{\text{func}}<0.20H_{\text{task}}.
$$

### 可视化

```text
p1_base_contamination_heatmap.svg
p1_task_func_cosine.svg
p1_conditioning_vs_p5_gap.svg
p1_channel_entropy_vs_gap.svg
```

---

## P2：Base-preserving subspace implementation

### 目标

实现 BPFS1-BPFS6。每个 candidate 必须证明 inactive 时严格等价于 LQ base。

### 必须记录

```text
candidate
basis_formula
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
task_channel_type
functional_channel_type
functional_init
lambda_inactive_exact_zero
analytic_value_stat
implemented
implementation_status
```

### 判断标准

Implementation pass:

```text
BPFS1-BPFS6 至少实现 4 个
每个实现 candidate 可 forward/backward/update smoke
functional inactive equivalence smoke pass
```

Strict PureKAN contract:

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

### 可视化

```text
p2_bpfs_implementation_matrix.svg
p2_inactive_equivalence_smoke.svg
```

---

## P3：Contract / gradcheck / base-equivalence audit

### 目标

确认 BPFS 是 strict PureKAN，且 inactive 时不改变 LQ base。

### 必须记录

```text
candidate
contract_pass
GradRelErrMax
GradCosMin
max_logit_diff_inactive
mean_logit_diff_inactive
max_param_diff_task_channel
functional_channel_zero_norm
pairwise_R2
local_bump_R2
basis_condition_number
```

### 判断标准

Grad pass:

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

Base-equivalence pass:

$$
\max_x\|z_{\text{BPFS-off}}-z_{\text{LQ}}\|_\infty\leq10^{-6}.
$$

Interaction pass:

$$
R^2_{\text{pairwise}}\geq0.95.
$$

If deliberately local:

$$
R^2_{\text{local-bump}}\geq0.95.
$$

### 可视化

```text
p3_contract_grad_base_equivalence.svg
p3_pairwise_local_fit.svg
p3_functional_zero_norm.svg
```

---

## P4：P4/P5 base preservation qualification

### 目标

functional inactive 时，BPFS 必须保住 LQ base 的 P4/P5。

### 设置

```text
functional_update = off
lambda(e) = 0
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
baseline = MLP-match
reference = LQ-t2-h256
```

### 必须记录

System:

```text
forward_q50
forward_q90
backward_q50
backward_q90
step_q50
step_q90
memory_compact
memory_conservative
extra_kernel_count
```

Task:

```text
dataset
seed
KAN_acc
MLP_match_acc
LQ_reference_acc
delta_vs_mlp
delta_vs_LQ
near_pass
CEp99
margin_p10
ECE
NLL
basis_entropy
functional_channel_entropy
lift_condition_number
effective_rank
```

### 判断标准

P4 pass:

$$
forward_{q90}\leq1.25,
$$

$$
backward_{q90}\leq1.50,
$$

$$
step_{q90}\leq1.50,
$$

$$
memory\leq1.05.
$$

P5 near-pass:

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Base preservation:

$$
|\Delta Acc_{\text{BPFS-off}}-\Delta Acc_{\text{LQ}}|\leq0.002.
$$

### 可视化

```text
p4_system_pareto.svg
p4_base_preservation_matrix.svg
p4_delta_vs_lq_by_candidate.svg
```

---

## P5：Functional carrier actuatability audit

### 目标

在 base-preserving candidate 上测试 functional subspace 是否能产生 non-silent movement，但仍不进入 paired replay。

### 设置

```text
functional_update = event-time only
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
events = signal strata S1-S8
horizons = 1,5,20,80,240
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
candidate
event_id
signal_stratum
horizon
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
```

### 判断标准

Carrier non-silent:

$$
r_{z,\text{tail}}\geq0.10.
$$

Non-AdamW component:

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety:

$$
BadEventRate\leq0.05.
$$

If carrier is silent:

```text
route = R2-BasePreservedButFunctionalSilent
```

### 可视化

```text
p5_functional_movement_distribution.svg
p5_tail_movement_vs_task_safety.svg
p5_nonadamw_component.svg
```

---

## P6：Value observability audit

### 目标

判断 BPFS 的 score 是否预测 grounded Real-vs-control value。

### 必须记录

```text
candidate
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
```

### 判断标准

Observability pass:

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept/abstain pass:

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Legality pass:

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

System pass:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_value_score_vs_grounded_value.svg
p6_auc_precision_coverage_by_bpfs.svg
p6_score_components_correlation.svg
p6_accepted_strata_distribution.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 BPFS controller 隐式 dataset tuning。

### 设置

Leave-dataset-out:

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out:

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
candidate
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
```

### 判断标准

LDO pass:

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two LDO splits satisfy:

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass:

At least $70\%$ held-out strata task-safe and CE-tail non-worse:

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

## P8：Official base-preserving functional paired replay

### 目标

只有 P4-P7 survivor 可以进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
candidate = best BPFS survivor
controller = best legal signal-based controller
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled
```

### 必须记录

```text
candidate
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
```

### 判断标准

Official paired replay pass:

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety:

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls fail:

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System gate:

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

验证 local causality 是否能在连续训练中保持。只有 P8 survivor 才能进入。

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
```

### 判断标准

Task safety:

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority:

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one:

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

## P10：Full 10-seed functional validation

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
```

### 判断标准

Task safety:

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement:

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair:

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority:

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one:

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

### 可视化

```text
p10_macro_delta_vs_controls.svg
p10_hard_stratum_repair_matrix.svg
p10_seedwise_win_matrix.svg
p10_task_geometry_pareto.svg
p10_ce_tail_margin_panel.svg
p10_functional_channel_usage_trace.svg
```

---

## P11：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。P11 并行执行，不使用 functional update，不使用 teacher/loss/sampler/class weight。

### Allowed repairs

```text
orthogonal lift init
fan-in output scale
centered / normalized T2
Legendre2-only
bounded rational base
piecewise local base
shared RBF base
dual-role basis with functional channel frozen during AdamW-only phase
hidden bracket h224/h256/h288
basis-balanced init
```

### 必须记录

```text
candidate
dataset
seed
test_acc
delta_vs_mlp
near_pass
full_pass
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
P4_step_q90
memory_ratio
```

### 判断标准

Full-pass:

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Robust near-pass:

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

### 可视化

```text
p11_adamw_fullpass_gap.svg
p11_hard_stratum_miss_rows.svg
p11_ce_tail_margin.svg
p11_basis_entropy_vs_gap.svg
```

---

## P12：Robustness and external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

Strong baselines:

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ/A7c AdamW-only
strict PureKAN functional candidate
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

Robustness pass:

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass:

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass:

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

### 可视化

```text
p12_noise_robustness_curve.svg
p12_strong_baseline_pareto.svg
p12_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9236.csv
p0_v9235_boundary_reproduction.csv
p1_vaop_failure_autopsy.csv
p2_bpfs_implementation.csv
p3_contract_grad_base_equivalence.csv
p4_bpfs_p4_p5_base_preservation.csv
p5_functional_carrier_actuatability.csv
p6_value_observability_audit.csv
p7_leave_dataset_and_stratum_out_validation.csv
p8_official_base_preserving_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
p11_adamw_only_fullpass_repair.csv
p12_robustness_external_ready.csv
bpfs_primitive_trace_v9236.csv
base_preservation_trace_v9236.csv
functional_carrier_trace_v9236.csv
value_score_trace_v9236.csv
leave_dataset_out_trace_v9236.csv
paired_replay_branch_trace_v9236.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9235_boundary_unstable
F3_dataset_tuning_detected
F4_vaop_failure_unattributed
F5_bpfs_not_implemented
F6_base_equivalence_fail
F7_bpfs_grad_fail
F8_bpfs_p4_fail
F9_bpfs_p5_fail
F10_base_preservation_fail
F11_functional_carrier_silent
F12_value_observability_fail
F13_value_score_system_too_expensive
F14_leave_dataset_out_fail
F15_leave_stratum_out_fail
F16_paired_replay_control_equivalent
F17_shuffle_control_pass
F18_functional_lr_equivalent
F19_short_run_task_drop
F20_full_run_no_macro_hard_stratum_gain
F21_adamw_fullpass_fail
F22_strong_baseline_explains_gain
F23_robustness_fail
F24_external_not_ready
F25_fake_or_proxy_violation
F26_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-VAOPFailureAttributed:
  v9.2.35 VAOP failure is attributed to base contamination / channel collision / conditioning / value mismatch.

R2-BasePreservingSubspaceImplemented:
  BPFS candidates implemented and inactive equivalence smoke passes.

R3-BPFSContractAndEquivalencePass:
  at least one BPFS passes contract / grad / base equivalence.

R4-BPFSBaseQualified:
  at least one BPFS passes P4 and P5 near-pass while inactive.

R5-FunctionalCarrierActive:
  functional subspace produces non-silent event-time movement without breaking task safety.

R6-BPFSValueObservabilityPass:
  BPFS value score predicts grounded Real-vs-control value.

R7-LeaveDatasetOutBPFSValuePass:
  BPFS controller generalizes without dataset-specific tuning.

R8-BasePreservingPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-BasePreservedButFunctionalSilent:
  BPFS preserves base but functional channel cannot move output enough.

R10-BasePreservedButValueUnobservable:
  BPFS preserves base and moves output, but score cannot predict value.

R11-BPFSAllFailPrimitiveReset:
  BPFS candidates fail base preservation / actuatability / observability.

R12-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R13-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R14-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R15-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9235_boundary_pass
dataset_tuning_detected
vaop_failure_mode
bpfs_implemented_count
best_bpfs_candidate
base_equivalence_pass
bpfs_contract_pass
bpfs_grad_pass
bpfs_p4_pass
bpfs_p5_nearpass
base_preservation_pass
functional_carrier_pass
value_observability_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
functional_task_safe
functional_control_pass
functional_system_pass
hard_stratum_repair_pass
adamw_fullpass
strong_baseline_pass
robustness_pass
external_ready
primary_blocker
next_required_implementation
success_v9236_strict_purekan_functional
success_v9236_full_functional
success_v9236_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.35 boundary。

Step 2:
  P1 做 VAOP failure autopsy。
  重点确认 P5 near-pass 失败是 base contamination、channel collision、conditioning 还是 value score 错误。

Step 3:
  P2 实现 BPFS1-BPFS6。
  不允许继续把 value channel 直接混入 base task channel。

Step 4:
  P3 做 contract / gradcheck / base-equivalence audit。
  inactive functional channel 必须严格等价 LQ base。

Step 5:
  P4 做 P4/P5 base preservation qualification。
  functional update off，不允许 functional update 救 base。

Step 6:
  P5 做 functional carrier actuatability。
  只测试 event-time functional channel 是否能动，不进入 paired replay。

Step 7:
  P6 做 value observability audit。
  重点看 control-gap score，而不是 movement 大小。

Step 8:
  P7 做 leave-dataset-out / leave-stratum-out。
  防止隐式 dataset tuning。

Step 9:
  P8 official paired replay。
  只有 BPFS survivor 才能进入。

Step 10:
  P9 short-run validation。

Step 11:
  P10 full 10-seed validation。

Step 12:
  P11 并行 AdamW-only full-pass repair。

Step 13:
  P12 robustness / strong baseline / external-ready。
```

---

## 9. 停止条件

### Minimum diagnostic success

```text
v9.2.35 boundary reproduced
VAOP P5 failure attributed
BPFS candidates implemented
at least one BPFS passes contract / grad / base equivalence
no fake/proxy/offload/loss/teacher violation
```

### Base-preserving functional carrier success

```text
Minimum diagnostic success
+
at least one BPFS passes P4/P5 base preservation
+
functional carrier non-silent
+
task safety holds
```

### Local functional success

```text
Base-preserving functional carrier success
+
value observability pass
+
leave-dataset-out pass
+
official paired replay beats AdamWParallel / bestLR
```

### Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

### External-ready success

```text
Full functional success
+
robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v9.2.35 boundary cannot be reproduced；
2. VAOP failure cannot be attributed；
3. BPFS candidates cannot be implemented as strict PureKAN；
4. all BPFS candidates fail base equivalence；
5. all BPFS candidates fail P4/P5 base preservation；
6. all BPFS candidates preserve base but functional channel is silent；
7. all BPFS candidates preserve base and move output but value score remains unobservable；
8. leave-dataset-out fails；
9. paired replay remains control-equivalent；
10. shuffle controls pass, indicating overfit；
11. full run gives no macro / hard-stratum / geometry gain；
12. functional breaks system gate；
13. gains are explained by QuadraticFeatureMLP；
14. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：BPFS restores P5

可以声明：

```text
v9.2.35 failed because value-aligned functional channel contaminated the base task channel.
```

但不能声明 functional success unless observability / paired replay pass.

### Case B：BPFS preserves base but functional is silent

必须声明：

```text
base preservation succeeded, but functional subspace lacks event-time actuatability.
```

下一步修 functional carrier，不是 dataset patch。

### Case C：BPFS preserves base and moves output but value score fails

必须声明：

```text
functional movement is possible, but still not control-resistant value.
```

下一步回到 value statistic / role-wise mechanism。

### Case D：BPFS passes observability but fails LDO

必须声明：

```text
controller is implicitly dataset-specific; official success is not allowed.
```

### Case E：BPFS paired replay passes

可以声明：

```text
strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

### Case F：all BPFS fail

必须声明：

```text
current strict FC-PureKAN functional carrier family cannot preserve base and expose value simultaneously.
```

下一步应回到 deeper edge-function interface design，尤其重新抽取 v8-FT7 role-wise mechanism。

---

## 11. 最终建议

v9.2.36 的一句话策略是：

$$
\boxed{
\text{不要再把 value-aligned primitive 直接塞进 base；先做 base-preserving functional subspace。}
}
$$

当前最关键的问题不是：

```text
Fashion 怎么调过；
KMNIST 怎么调过；
MNIST 是否要 abstain；
再换哪个 target；
再调哪个 horizon；
```

而是：

```text
1. VAOP 为什么破坏 P5 near-pass？
2. functional value channel 能否在 inactive 时严格不影响 LQ base？
3. functional channel active 后是否能产生 non-silent, non-AdamW movement？
4. control-gap score 是否能预测 grounded Real-vs-control value？
5. 这个规则能否 leave-dataset-out 泛化？
6. RealFunctional 能否在 official paired replay 中超过 AdamWParallel / best LR？
7. 如果 BPFS 也失败，是否需要重新从 v8-FT7 role-wise mechanism 设计 strict edge-owned functional interface？
```
