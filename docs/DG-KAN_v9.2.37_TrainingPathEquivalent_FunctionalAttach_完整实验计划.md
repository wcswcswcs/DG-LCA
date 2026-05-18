# DG-KAN v9.2.37 Training-Path Equivalent Functional Attach：从 BPFS base-equivalence smoke 到训练轨迹等价 functional carrier 的完整实验计划

> 本计划基于 v9.2.36 `Base-Preserving Value-Aligned Functional Subspace` 的真实复盘制定。  
> v9.2.36 的 terminal route 是：
>
> ```text
> route = R11-BPFSAllFailPrimitiveReset
> base_candidate = LQ-t2-h256
> success_v9236_strict_purekan_functional = False
> success_v9236_full_functional = False
> success_v9236_external_ready = False
> ```
>
> v9.2.36 的关键事实是：**BPFS1-BPFS6 已经实现，contract / grad / 初始 base-equivalence 都通过，但 route-level base qualification 没有通过；functional carrier 没有打开，max $r_{z,\text{tail}}=0$，max $r_{\perp,\text{tail}}=0$。**
>
> 这说明 v9.2.36 没有证明 functional update 失败。它证明的是：
>
> $$
> \boxed{
> \text{“初始 inactive 等价”不等于“训练轨迹等价”。}
> }
> $$
>
> 因此 v9.2.37 的核心任务不是再设计一个更复杂的 value score，也不是针对 Fashion / KMNIST / MNIST 调参，而是把 base preservation 从 logit-smoke 升级为 **training-path equivalence**。  
> 只有当 dormant / late-attached functional carrier 在 base phase 不改变 LQ/T2 的初始化、梯度、optimizer state、minibatch 轨迹和 AdamW-only P5 表现时，才允许测试它的 functional actuatability 与 value causality。

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
  report signal-stratum composition by dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
```

---

## 1. v9.2.36 独立判断

### 1.1 v9.2.36 没有达到目标

v9.2.36 的核心结果是：

```text
P0:
  v9.2.35 boundary reproduced
  source route = R9-VAOPAllFailPrimitiveReset
  source primary blocker = all_vaop_failed_P5_nearpass

P1:
  VAOP failure mode = base_contamination
  base contamination rate = 1.0

P2:
  BPFS implemented count = 6

P3:
  base equivalence / contract / grad = 1 / 1 / 1

P4:
  route-level BPFS base qualification = 0
  P4 pass = 1
  P5 near-pass = 0
  best BPFS = empty

P5:
  carrier pass = 0
  max r_z_tail = 0.000000
  max r_perp_tail = 0.000000

route:
  R11-BPFSAllFailPrimitiveReset
  blocker = no_bpfs_passed_P4_P5_base_qualification
```

所以当前不能声明：

```text
strict PureKAN functional success
base-preserving functional carrier success
value observability success
paired replay success
short-run / full-run / external-ready
```

### 1.2 这不是 functional update 的失败结论

P5 carrier 的 $r_z=0$ 不能被解释为 “functional channel 本质上不能动”。  
原因是：P4/P5 base qualification 没有打开 route-level survivor，functional carrier 按 gate 不应进入正式 actuation。也就是说，P5 的 $r_z=0$ 更像 **gate-blocked dormant carrier**，不是一次合法的 functional causality failure。

### 1.3 v9.2.36 真正推进了什么

v9.2.35 说明 VAOP 直接混入 base 后 P5 near-pass 全失败；v9.2.36 说明 BPFS 做到了：

```text
implementation pass
contract pass
grad pass
initial inactive equivalence pass
```

这已经是明显推进。  
它把问题从：

```text
functional value channel 直接污染 base
```

推进到：

```text
functional channel 初始可 inactive，但训练轨迹仍没有保住 base qualification
```

所以真正 blocker 变成：

$$
\boxed{
\text{training-path preservation 没有闭合。}
}
$$

### 1.4 当前不能继续做什么

v9.2.37 不能继续做下面这些小修补：

```text
1. 再调 BPFS value score；
2. 再调 lambda threshold；
3. 再调 Fashion / KMNIST 特化 event；
4. 直接 short-run / full-run；
5. 把 base-equivalence smoke 当作 base preservation success；
6. 把 functional carrier r_z=0 当成 functional concept failure；
7. 用 functional update 去救 P5 failed base。
```

---

## 2. 当前问题的本质

### 2.1 Base-equivalence smoke 太弱

v9.2.36 的 base equivalence pass 只说明：

$$
\max_x
\|z_{\text{BPFS-off}}(x)-z_{\text{LQ}}(x)\|_\infty
\leq 10^{-6}
$$

在某个初始或 smoke 条件下成立。  
但 P5 near-pass 失败说明，训练过程中的某些因素仍在改变 base trajectory。

这些因素可能包括：

```text
1. functional params 的创建改变了 task-channel RNG 初始化；
2. functional params 注册到 optimizer 后改变 foreach / fused update path；
3. dormant params 产生 AdamW state 或 memory / timing 污染；
4. lambda=0 只在 forward smoke 中成立，但训练 loop 的某些 path 仍使用 functional channel；
5. task channel 与 functional channel 的参数 group / state dict / flatten order 改变了 update sequence；
6. MLP-match baseline 因 dormant params 增加而被错误匹配；
7. P4 pass 的 candidate 与 P5 near-pass 的 candidate 不是同一个稳定 survivor；
8. base preservation 只测了 logits，没有测 task gradients / optimizer states / per-step parameter drift。
```

### 2.2 现在要证明的是训练轨迹等价

v9.2.37 的关键不是“模型一开始相等”，而是：

$$
\boxed{
\text{当 functional inactive 时，BPFS 的训练轨迹必须等价于 LQ base。}
}
$$

具体包括：

$$
\theta^T_{0,\text{BPFS}} = \theta^T_{0,\text{LQ}},
$$

$$
z_{t,\text{BPFS-off}}(x)=z_{t,\text{LQ}}(x),
$$

$$
g^T_{t,\text{BPFS-off}}=g^T_{t,\text{LQ}},
$$

$$
m^T_{t,\text{BPFS-off}}=m^T_{t,\text{LQ}},
$$

$$
v^T_{t,\text{BPFS-off}}=v^T_{t,\text{LQ}},
$$

for the same minibatch sequence and task-channel parameter order.

### 2.3 functional channel 应该 late attach 或 optimizer-excluded dormant

如果 dormant functional params 在 base phase 仍然进入 optimizer/state/memory/param matching，就可能污染 base。  
因此 v9.2.37 要测试两类方案：

```text
A. registered dormant:
   functional params 注册在模型里，但不进入 optimizer，不进入 param-count matching，不进入 update path。

B. late attach:
   base phase 完全训练 LQ/T2；
   only after base qualification, attach zero-init functional edge channel；
   attach 后 inactive 时仍严格等价 base；
   active 时才进入 functional update rule。
```

这不是 external residual。attach 后仍是 edge-owned PureKAN function：

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T)
+
\lambda(e)\phi^{F}_{ij}(h;\theta_F).
$$

关键是 $\theta_F$ 在 base phase 不改变 $\theta_T$ 的训练轨迹。

---

## 3. v9.2.37 总体目标

v9.2.37 的总体目标是：

$$
\boxed{
\text{建立 training-path equivalent functional attach，使 LQ/T2 base 不被污染，同时保留 event-time functional actuatability。}
}
$$

这个目标分为五层。

### 3.1 Training-path equivalence success

Functional inactive 时，BPFS / TPEA 与 LQ base 在训练路径上等价：

$$
\max_{t\leq T}
\|z_{t,\text{TPEA-off}}-z_{t,\text{LQ}}\|_\infty
\leq 10^{-5}.
$$

Task-channel parameter drift：

$$
\max_{t\leq T}
\|\theta^T_{t,\text{TPEA-off}}-\theta^T_{t,\text{LQ}}\|_\infty
\leq 10^{-6}.
$$

Task-channel gradient drift：

$$
\max_{t\leq T}
\|g^T_{t,\text{TPEA-off}}-g^T_{t,\text{LQ}}\|_\infty
\leq 10^{-6}.
$$

Optimizer state drift：

$$
\max_{t\leq T}
\|m^T_{t,\text{TPEA-off}}-m^T_{t,\text{LQ}}\|_\infty
\leq 10^{-6},
$$

$$
\max_{t\leq T}
\|v^T_{t,\text{TPEA-off}}-v^T_{t,\text{LQ}}\|_\infty
\leq 10^{-6}.
$$

### 3.2 Base qualification success

Functional inactive 时必须恢复 LQ/T2 base qualification：

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

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

与 LQ reference 的差异：

$$
|\Delta Acc_{\text{TPEA-off}}-\Delta Acc_{\text{LQ}}|
\leq0.002.
$$

### 3.3 Functional carrier success

在 base-qualified TPEA 上，active event-time functional channel 必须 non-silent：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

### 3.4 Value observability success

Event-time score 必须预测 grounded Real-vs-control value：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

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

### 3.5 Local functional causality success

Only after training-path equivalence, base qualification, carrier actuatability, value observability, and leave-dataset-out pass, official paired replay can open.

Paired replay success：

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

## 4. 核心假设

### H1：v9.2.36 失败主要是训练路径污染，而不是 BPFS 概念错误

v9.2.36 已经做到 base equivalence / contract / grad pass，但 P5 near-pass fail。H1 认为问题不是 dormant functional channel 在数学上不可能，而是 training path 中还有污染源。

H1 成立标准：

若 P1 发现以下至少一项成立，则 H1 成立：

```text
task init hash drift
minibatch trace drift
task gradient drift
optimizer state drift
param-group / foreach path drift
functional params counted into MLP-match baseline
lambda inactive leak
```

并且 TPEA 修复后恢复 P5 near-pass。

### H2：late attach 比 registered dormant 更可能保住 base

Registered dormant params 即使 forward inactive，也可能影响 optimizer state、memory、foreach ordering、param count、baseline matching。  
Late attach 在 base phase 完全不创建 functional params，理论上最干净。

H2 成立标准：

如果：

$$
PathDrift_{\text{late-attach}}<PathDrift_{\text{registered-dormant}},
$$

且 late-attach P5 near-pass 恢复，则 H2 成立。

### H3：functional carrier 不能在 base phase 学习，只能在 event-time update

v9.2.35 说明 value-aligned channel 常驻训练会破坏 base。H3 认为 functional channel 应该：

```text
zero-init
base phase absent or optimizer-excluded
event-time only update
event-time only active
```

H3 成立标准：

TPEA-off 与 LQ base 等价，TPEA-on 产生 non-silent functional movement，且 task safety pass。

### H4：control-gap score 仍是最优先 value statistic

v9.2.35 的 OP failure attribution 显示：

```text
corr_control_gap = 0.288810
corr_r_perp_tail = -0.031948
corr_obs_score = 0.000848
```

因此 v9.2.37 不再优先看 movement-only score，而是继续直接估计：

$$
S_{\text{gap}}
=
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

H4 成立标准：

TPEA 的 control-gap score 比 movement-only score 更能预测 grounded value：

$$
AUC(S_{\text{gap}})-AUC(r_{\perp,\text{tail}})\geq0.10
$$

or:

$$
Corr(S_{\text{gap}},V)-Corr(r_{\perp,\text{tail}},V)\geq0.10.
$$

### H5：如果 training-path equivalence 仍不能恢复 P5，说明当前 BPFS attach 机制不合格

如果 late attach / optimizer-excluded dormant / deterministic clone 都无法恢复 LQ base P5，说明当前 BPFS implementation 仍然改变 task base，必须先修 training infrastructure，而不是继续 functional research。

### H6：如果 base 保住但 functional carrier silent，问题转到 actuatability

如果 TPEA-off 过 P4/P5，但 TPEA-on 仍有：

$$
r_{z,\text{tail}}<0.10,
$$

则 route 应是：

```text
BasePreservedButFunctionalSilent
```

下一步修 functional carrier，不是 dataset patch。

### H7：如果 carrier 能动但 value 不过，才回到 value statistic / role-wise mechanism

如果 TPEA-on 满足 carrier movement，但 value observability / paired replay 失败，说明 base preservation 已解决，blocker 转到 functional value alignment。此时才应回到 FT7 role-wise mechanism 或更强 value statistic。

---

## 5. Candidate 设计

## 5.1 Baselines

```text
LQ0-LQ-t2-h256:
  current strict FC-PureKAN AdamW-only base。

BPFS-v9236-best:
  v9.2.36 BPFS family reference。

VAOP6-current-best-P4:
  v9.2.35 best measured VAOP, P4 pass but P5 fail。

V8-FT7:
  mechanism source only, not official strict candidate。
```

## 5.2 Training-path equivalent attach candidates

### TPEA0：LQ Base Reference

No functional channel.  
Used for exact path comparison.

### TPEA1：Registered Zero Functional Excluded From Optimizer

Functional params exist in model, but:

```text
lambda = 0
functional params zero-init
functional params not included in AdamW optimizer
functional params excluded from param-count MLP matching
functional params excluded from foreach/fused update list
```

### TPEA2：Registered Zero Functional Separate NoOp Optimizer Group

Functional params exist, but optimizer group has:

```text
lr = 0
weight_decay = 0
betas irrelevant
state allocation disabled if possible
```

Purpose: determine whether param registration alone breaks path.

### TPEA3：Late Attach After LQ Base Warmup

Phase A:

```text
train LQ base normally
save task params and optimizer state
```

Phase B:

```text
attach zero-init functional channel
do not modify task optimizer state
lambda = 0 until event
```

Inactive equivalence must hold after attach.

### TPEA4：Shadow Functional Spec Not Registered Until Event

Functional channel exists only as a symbolic / spec object until an event passes gate.  
When event triggers, instantiate edge-owned functional params with zero init and apply one functional update.

### TPEA5：RoleWise FT7 Late-Attach Edge-Owned Carrier

Late-attached role-wise functional channel:

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T)
+
\lambda(e)
[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
].
$$

Role weights are signal-based, not dataset-based.

### TPEA6：ControlGap Late-Attach Carrier

Late-attached functional channel accepts only if:

$$
LCB(Gain_{\text{Real}})
>
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

This is conservative and may have low coverage, but it tests whether a clean base-preserving carrier can produce high precision.

---

## 6. 实验阶段

## P0：v9.2.36 boundary reproduction

### 目标

复现 v9.2.36 boundary，确认当前问题不是 measurement artifact。

### 必须记录

```text
route
source_route_v9235
vaop_failure_mode
base_contamination_rate
bpfs_implemented_count
base_equivalence_pass
bpfs_contract_pass
bpfs_grad_pass
bpfs_p4_pass
bpfs_p5_nearpass
base_preservation_pass
functional_carrier_pass
max_r_z_tail
max_r_perp_tail
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R11-BPFSAllFailPrimitiveReset
base equivalence / contract / grad = 1 / 1 / 1
P5 near-pass = 0
functional carrier pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_bpfs_gate_ladder.svg
p0_base_equivalence_vs_p5_failure.svg
```

---

## P1：BPFS training-path failure autopsy

### 目标

解释为什么 base equivalence pass，但 P5 near-pass fail。  
P1 是 v9.2.37 的关键诊断，不能跳过。

### 设置

```text
candidates = BPFS1-BPFS6, TPEA0-LQ-reference
functional_update = off
lambda = 0
steps = 0,1,5,20,100,full-epoch sample
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### 必须记录

```text
candidate
dataset
seed
step
task_init_hash_match
task_param_order_match
task_param_shape_match
functional_param_registered
functional_param_in_optimizer
functional_param_in_param_count
functional_optimizer_state_allocated
minibatch_indices_hash
logit_max_diff_vs_LQ
logit_mean_diff_vs_LQ
task_grad_max_diff_vs_LQ
task_param_max_diff_vs_LQ
adamw_m_max_diff_vs_LQ
adamw_v_max_diff_vs_LQ
functional_param_norm
functional_grad_norm
lambda_value
lambda_leak_detected
foreach_group_signature
optimizer_group_count
MLP_match_param_count
KAN_param_count_official
```

### 判断标准

Training-path equivalence pass：

$$
\max_t\|z_{t,\text{candidate}}-z_{t,\text{LQ}}\|_\infty\leq10^{-5},
$$

$$
\max_t\|\theta^T_{t,\text{candidate}}-\theta^T_{t,\text{LQ}}\|_\infty\leq10^{-6},
$$

$$
\max_t\|g^T_{t,\text{candidate}}-g^T_{t,\text{LQ}}\|_\infty\leq10^{-6}.
$$

Failure attribution pass：

```text
>= 90% failed rows assigned to one primary failure:
  RNG/init drift
  optimizer-state pollution
  lambda leak
  param-count / MLP-match drift
  foreach/update-order drift
  functional-grad leak
  candidate mismatch P4-vs-P5
  other measured reason
```

### 可视化

```text
p1_path_drift_over_steps.svg
p1_optimizer_state_drift.svg
p1_param_group_signature.svg
p1_failure_mode_sankey.svg
```

---

## P2：Training-path equivalent attach implementation

### 目标

实现 TPEA1-TPEA6，保证 inactive 时不会改变 LQ base 训练轨迹。

### 必须记录

```text
candidate
attach_mode
basis_formula
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
functional_param_registered_base_phase
functional_param_optimizer_base_phase
functional_param_counted_base_phase
lambda_inactive_exact_zero
task_channel_clone_source
optimizer_state_clone_source
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
TPEA1-TPEA6 至少实现 4 个
TPEA3 或 TPEA4 必须实现至少一个
所有 implemented candidates forward/backward/update smoke pass
no external residual / ordinary MLP path
```

Strict PureKAN contract：

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
p2_tpea_implementation_matrix.svg
p2_attach_mode_comparison.svg
```

---

## P3：Contract / gradcheck / training-path equivalence audit

### 目标

确认 TPEA 是 strict PureKAN，并且 inactive 时训练轨迹等价。

### 必须记录

```text
candidate
contract_pass
GradRelErrMax
GradCosMin
max_logit_diff_inactive
mean_logit_diff_inactive
max_task_param_diff
max_task_grad_diff
max_adamw_m_diff
max_adamw_v_diff
functional_zero_norm
functional_grad_zero_norm
pairwise_R2
local_bump_R2
basis_condition_number
```

### 判断标准

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

Base path equivalence pass：

$$
\max_t\|z_{t,\text{TPEA-off}}-z_{t,\text{LQ}}\|_\infty\leq10^{-5},
$$

$$
\max_t\|\theta^T_{t,\text{TPEA-off}}-\theta^T_{t,\text{LQ}}\|_\infty\leq10^{-6}.
$$

Interaction pass：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

If deliberately local：

$$
R^2_{\text{local-bump}}\geq0.95.
$$

### 可视化

```text
p3_contract_grad_path_equivalence.svg
p3_logit_diff_trace.svg
p3_task_param_diff_trace.svg
p3_pairwise_local_fit.svg
```

---

## P4：P4/P5 base preservation qualification

### 目标

functional inactive 时，TPEA 必须恢复 LQ base P4/P5。

### 设置

```text
functional_update = off
lambda = 0
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
baseline = MLP-match
reference = LQ-t2-h256
```

### 必须记录

System：

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
functional_param_memory_in_base_phase
optimizer_state_memory_base_phase
```

Task：

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

P4 pass：

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

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Base preservation：

$$
|\Delta Acc_{\text{TPEA-off}}-\Delta Acc_{\text{LQ}}|\leq0.002.
$$

### 可视化

```text
p4_system_pareto.svg
p4_base_preservation_matrix.svg
p4_delta_vs_lq_by_tpea.svg
```

---

## P5：Functional carrier actuatability audit

### 目标

只在 P4 survivor 上测试 event-time functional carrier 是否能动。  
P5 不是 paired replay，也不是 task success。

### 设置

```text
candidates = P4 survivors only
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

If P4 survivor exists but carrier silent：

```text
route = R9-BasePreservedButFunctionalSilent
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

判断 TPEA 的 value score 是否预测 grounded Real-vs-control value。

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

Legality pass：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_value_score_vs_grounded_value.svg
p6_auc_precision_coverage_by_tpea.svg
p6_score_components_correlation.svg
p6_accepted_strata_distribution.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 TPEA controller 隐式 dataset tuning。

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

## P8：Official training-path-equivalent paired replay

### 目标

只有 P4-P7 survivor 可以进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
candidate = best TPEA survivor
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

System gate：

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

Mechanism pass, at least one：

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

Mechanism pass, at least one：

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

Full-pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Robust near-pass：

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

Strong baselines：

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

### 可视化

```text
p12_noise_robustness_curve.svg
p12_strong_baseline_pareto.svg
p12_external_ready_scorecard.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9237.csv
p0_v9236_boundary_reproduction.csv
p1_bpfs_training_path_failure_autopsy.csv
p2_tpea_implementation.csv
p3_contract_grad_training_path_equivalence.csv
p4_tpea_p4_p5_base_preservation.csv
p5_functional_carrier_actuatability.csv
p6_value_observability_audit.csv
p7_leave_dataset_and_stratum_out_validation.csv
p8_official_training_path_equivalent_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
p11_adamw_only_fullpass_repair.csv
p12_robustness_external_ready.csv
training_path_trace_v9237.csv
optimizer_state_trace_v9237.csv
attach_mode_trace_v9237.csv
functional_carrier_trace_v9237.csv
value_score_trace_v9237.csv
leave_dataset_out_trace_v9237.csv
paired_replay_branch_trace_v9237.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9236_boundary_unstable
F3_dataset_tuning_detected
F4_bpfs_failure_unattributed
F5_tpea_not_implemented
F6_task_init_hash_drift
F7_training_path_equivalence_fail
F8_optimizer_state_pollution
F9_lambda_inactive_leak
F10_param_count_or_mlp_match_drift
F11_tpea_grad_fail
F12_tpea_p4_fail
F13_tpea_p5_fail
F14_base_preservation_fail
F15_functional_carrier_silent
F16_value_observability_fail
F17_value_score_system_too_expensive
F18_leave_dataset_out_fail
F19_leave_stratum_out_fail
F20_paired_replay_control_equivalent
F21_shuffle_control_pass
F22_functional_lr_equivalent
F23_short_run_task_drop
F24_full_run_no_macro_hard_stratum_gain
F25_adamw_fullpass_fail
F26_strong_baseline_explains_gain
F27_robustness_fail
F28_external_not_ready
F29_fake_or_proxy_violation
F30_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-BPFSFailureAttributed:
  v9.2.36 BPFS failure attributed to training-path drift / optimizer pollution / lambda leak / param-count drift.

R2-TPEAImplemented:
  TPEA candidates implemented with strict PureKAN contract.

R3-TPEATrainingPathEquivalent:
  at least one TPEA candidate matches LQ base training path when inactive.

R4-TPEABaseQualified:
  at least one TPEA passes P4/P5 base preservation.

R5-FunctionalCarrierActive:
  event-time functional subspace produces non-silent, non-AdamW movement.

R6-TPEAValueObservabilityPass:
  value score predicts grounded Real-vs-control value.

R7-LeaveDatasetOutTPEAPass:
  controller generalizes without dataset-specific tuning.

R8-TrainingPathEquivalentPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-BasePreservedButFunctionalSilent:
  base preservation succeeds but functional carrier cannot move output enough.

R10-BasePreservedButValueUnobservable:
  base preservation and movement succeed, but score cannot predict value.

R11-TPEAAllFailPrimitiveReset:
  TPEA candidates fail path equivalence / base preservation / actuatability / observability.

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
v9236_boundary_pass
dataset_tuning_detected
bpfs_failure_mode
tpea_implemented_count
best_tpea_candidate
training_path_equivalence_pass
task_init_hash_match
optimizer_state_match
lambda_inactive_leak_detected
tpea_contract_pass
tpea_grad_pass
tpea_p4_pass
tpea_p5_nearpass
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
success_v9237_strict_purekan_functional
success_v9237_full_functional
success_v9237_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.36 boundary。

Step 2:
  P1 做 BPFS training-path failure autopsy。
  不要直接实现新 value score；先查清 base equivalence pass 为什么没有转成 P5 pass。

Step 3:
  P2 实现 TPEA1-TPEA6。
  至少实现一个 late attach 和一个 optimizer-excluded dormant 方案。

Step 4:
  P3 做 contract / gradcheck / training-path equivalence。
  inactive functional channel 必须匹配 LQ base 的训练轨迹，而不只是初始 logits。

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
  只有 TPEA survivor 才能进入。

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

## 10. 停止条件

### Minimum diagnostic success

```text
v9.2.36 boundary reproduced
BPFS P5 failure attributed
TPEA candidates implemented
at least one TPEA passes contract / grad / training-path equivalence
no fake/proxy/offload/loss/teacher violation
```

### Base-preserving functional carrier success

```text
Minimum diagnostic success
+
at least one TPEA passes P4/P5 base preservation
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
1. v9.2.36 boundary cannot be reproduced；
2. BPFS failure cannot be attributed；
3. TPEA candidates cannot be implemented as strict PureKAN；
4. all TPEA candidates fail training-path equivalence；
5. all TPEA candidates fail P4/P5 base preservation；
6. all TPEA candidates preserve base but functional channel is silent；
7. all TPEA candidates preserve base and move output but value score remains unobservable；
8. leave-dataset-out fails；
9. paired replay remains control-equivalent；
10. shuffle controls pass, indicating overfit；
11. full run gives no macro / hard-stratum / geometry gain；
12. functional breaks system gate；
13. gains are explained by QuadraticFeatureMLP；
14. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：TPEA restores P5

可以声明：

```text
v9.2.36 failed because initial base equivalence did not imply training-path equivalence; TPEA restored base preservation.
```

但不能声明 functional success unless actuatability / observability / paired replay pass.

### Case B：TPEA preserves base but functional is silent

必须声明：

```text
base preservation succeeded, but functional carrier lacks event-time actuatability.
```

下一步修 functional carrier，不是 dataset patch。

### Case C：TPEA preserves base and moves output but value score fails

必须声明：

```text
functional movement is possible, but still not control-resistant value.
```

下一步回到 value statistic / role-wise mechanism。

### Case D：TPEA passes observability but fails LDO

必须声明：

```text
controller is implicitly dataset-specific; official success is not allowed.
```

### Case E：TPEA paired replay passes

可以声明：

```text
strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

### Case F：all TPEA fail

必须声明：

```text
current strict FC-PureKAN functional carrier family cannot preserve base training path and expose value simultaneously.
```

下一步应回到 deeper edge-function interface design，尤其重新抽取 v8-FT7 role-wise mechanism。

---

## 12. 最终建议

v9.2.37 的一句话策略是：

$$
\boxed{
\text{不要再满足于 base-equivalence smoke；必须证明 functional inactive 时训练轨迹等价于 LQ base。}
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
1. BPFS 为什么 initial equivalence pass 但 P5 near-pass fail？
2. functional params 是否污染了 RNG、optimizer state、param group、baseline matching 或 update path？
3. late attach / optimizer-excluded dormant 是否能恢复 LQ base trajectory？
4. base preserved 后 functional carrier 是否能产生 non-silent, non-AdamW movement？
5. control-gap score 是否能预测 grounded Real-vs-control value？
6. 这个规则能否 leave-dataset-out 泛化？
7. RealFunctional 能否在 official paired replay 中超过 AdamWParallel / best LR？
8. 如果 TPEA 也失败，是否需要重新从 v8-FT7 role-wise mechanism 设计 strict edge-owned functional interface？
```
