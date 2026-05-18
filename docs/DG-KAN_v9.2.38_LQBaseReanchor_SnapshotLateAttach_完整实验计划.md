# DG-KAN v9.2.38 LQ Base Re-Anchor 与 Snapshot Late-Attach Functional Carrier 完整实验计划

> 本计划基于 v9.2.37 `Training-Path Equivalent Functional Attach` 的真实复盘结果制定。  
> v9.2.37 的 terminal route 是：
>
> ```text
> route = R11-TPEAAllFailPrimitiveReset
> base_candidate = LQ-t2-h256
> success_v9237_strict_purekan_functional = False
> success_v9237_full_functional = False
> success_v9237_external_ready = False
> ```
>
> 本轮最关键的事实不是 “functional update 失败”，而是：
>
> $$
> \boxed{
> \text{TPEA 已经做到 training-path equivalence / contract / grad pass，但 P5 仍没有 near-pass。}
> }
> $$
>
> 这说明当前 blocker 已经不是简单的 `functional channel contaminates base path`。如果 inactive TPEA 与 LQ base 的训练路径已经等价，但 P5 仍失败，那么问题必须被重新定位为：
>
> $$
> \boxed{
> \text{LQ base re-anchor / P5 protocol / candidate identity / MLP-match definition 尚未闭合。}
> }
> $$
>
> 因此 v9.2.38 不应继续设计更复杂的 functional value score，也不应针对 Fashion / KMNIST / MNIST 调参。  
> v9.2.38 的核心任务是：**先把当前代码路径下的 LQ base 重新锚定为可复现 P4/P5 reference，再以 checkpoint-level snapshot late-attach 的方式测试 functional carrier。**

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

# Part I. 对 v9.2.37 的独立判断

## 1. v9.2.37 没有达到目标

v9.2.37 的核心结果是：

```text
P0:
  v9.2.36 boundary reproduced
  source route = R11-BPFSAllFailPrimitiveReset
  source blocker = no_bpfs_passed_P4_P5_base_qualification

P1:
  BPFS failure mode = candidate_mismatch_or_LQ_base_gap
  failure attribution fraction = 1.0

P2:
  TPEA implemented count = 6

P3:
  training-path equivalence / contract / grad = 1 / 1 / 1

P4:
  best TPEA = TPEA1-RegisteredZeroExcluded
  P4 pass = 1
  P5 near-pass = 0
  base preservation = 0

P5:
  carrier pass = 0
  max r_z_tail = 0.000000
  max r_perp_tail = 0.000000

route:
  R11-TPEAAllFailPrimitiveReset
  blocker = all_tpea_failed_P4_P5_base_preservation
```

因此不能声明：

```text
strict PureKAN functional success
base-preserving functional carrier success
value observability success
paired replay success
short-run / full-run / external-ready
```

## 2. v9.2.37 的推进在哪里

v9.2.37 并不是原地失败。它完成了三件重要事情。

第一，TPEA1-TPEA6 已经实现，而不是停留在未实现状态。

第二，P3 已经做到：

```text
training-path equivalence pass = 1
contract pass = 1
grad pass = 1
```

这比 v9.2.36 的 “initial base equivalence smoke” 更强，说明 `functional inactive` 路径至少在 P3 审计范围内没有明显破坏 task-channel path。

第三，P1 把 BPFS 失败重新归因为：

```text
candidate_mismatch_or_LQ_base_gap
```

这很关键。它意味着：**当前 P5 failure 很可能不是 functional channel 导致的新污染，而是 LQ reference / P5 reproduction / candidate identity / MLP-match protocol 本身没有对齐。**

## 3. 当前不能怎样解释

不能解释为：

```text
functional update 已经失败；
TPEA 方向完全失败；
LQ/T2 base 应该放弃；
应该针对 Fashion / KMNIST 写特殊 route；
应该直接进入 short/full run；
应该继续调 value score。
```

P5 carrier 的：

```text
max r_z_tail = 0
max r_perp_tail = 0
```

不能被当作 functional causality failure，因为 P4/P5 base-preservation gate 没有打开，functional carrier 按计划不应该进入正式 actuation。  
它更像是 gate-blocked dormant carrier，而不是一次合法 functional update 实验。

## 4. 当前真正 blocker

v9.2.37 之后，当前最准确的 blocker 是：

$$
\boxed{
\text{LQ base reference 与 TPEA-off / P5 protocol 没有形成同一条可复现锚点。}
}
$$

更具体地说，必须先回答：

```text
1. 当前 runner 下 LQ-t2-h256 本身是否仍能复现 v9.2.6 / v9.2.7 的 near-pass？
2. TPEA-off 是否和当前 LQ reference 在 P5 full training 而非短 path audit 中完全一致？
3. P5 near-pass 失败是因为 LQ base gap，还是因为 MLP-match baseline / param-count / seed / data split / hidden scale / fan-in output scale 发生了 drift？
4. 如果 LQ base 本身失败，functional carrier 不应继续测试。
5. 如果 LQ base 通过但 TPEA-off 失败，才说明 attach infrastructure 仍污染 base。
```

---

# Part II. v9.2.38 总体目标

v9.2.38 的总体目标是：

$$
\boxed{
\text{重新锚定 LQ/T2 base，并用 checkpoint-level snapshot late-attach 测试 functional carrier。}
}
$$

这个目标分成五层。

## 1. LQ base re-anchor success

当前代码路径下必须重新复现 LQ base reference：

$$
near\_pass\_rate_{\text{LQ}}\geq0.80,
$$

$$
\Delta Acc_{\text{macro,LQ}}\geq-0.01.
$$

并且至少与历史 v9.2.6 / v9.2.7 reference 在同一 tolerance 内：

$$
|\Delta Acc_{\text{macro,current LQ}}-\Delta Acc_{\text{macro,historical LQ}}|\leq0.003.
$$

如果当前 LQ base 本身无法复现，则 route 必须写成：

```text
R1-LQBaseAnchorBroken
```

此时不允许继续 functional update。

## 2. TPEA-off equivalence success

TPEA-off 与同一 runner 中当前 LQ reference 必须在完整 P5 training path 上等价，而不仅是短步 trace 等价：

$$
\max_{t\leq T}
\|z_{t,\text{TPEA-off}}-z_{t,\text{LQ}}\|_\infty
\leq10^{-5}.
$$

Task-channel parameter drift：

$$
\max_{t\leq T}
\|\theta^T_{t,\text{TPEA-off}}-\theta^T_{t,\text{LQ}}\|_\infty
\leq10^{-6}.
$$

P5 task equivalence：

$$
|\Delta Acc_{\text{TPEA-off}}-\Delta Acc_{\text{LQ}}|\leq0.001.
$$

## 3. Snapshot late-attach success

Functional channel 不再参与 base phase。先训练 LQ base，再从同一 checkpoint attach zero-init edge-owned functional carrier。

Base checkpoint：

$$
\theta_T^\star
=
Train_{\text{AdamW}}(\theta_T^0).
$$

Attach 后 inactive：

$$
z_{\text{late-attach-off}}(x;\theta_T^\star,\theta_F=0)
=
z_{\text{LQ}}(x;\theta_T^\star).
$$

要求：

$$
\max_x
\|z_{\text{late-attach-off}}-z_{\text{LQ checkpoint}}\|_\infty
\leq10^{-6}.
$$

并且：

```text
task params unchanged
optimizer state unchanged
functional params zero-init
functional params edge-owned
external residual = 0
ordinary MLP path = 0
```

## 4. Functional carrier success

只有 LQ anchor 与 snapshot attach 都通过后，才测试 active functional carrier。

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

## 5. Local functional causality success

Value score 必须预测 grounded Real-vs-control value：

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

Official paired replay：

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

## H1：v9.2.37 的 P5 failure 主要来自 LQ base anchor / candidate protocol mismatch

v9.2.37 已经通过 training-path equivalence / contract / grad，但 P5 near-pass 仍失败。  
如果 TPEA-off 和 LQ 真正路径等价，那么 TPEA-off P5 fail 应该意味着 LQ reference 在当前 runner 下也 fail，或者 P5 gate 的 reference 不再是历史 LQ。

H1 成立标准：

```text
current LQ P5 near-pass = 0
or historical-current LQ macro delta drift > 0.003
or MLP-match / param-count / hidden / seed / data split / fan-in scale mismatch detected
```

## H2：如果当前 LQ base re-anchor 成功，TPEA-off 也应该 P5 成功

若 LQ base 在同一 runner 下 P5 near-pass 成立，而 TPEA-off 仍失败，则说明 TPEA infrastructure 仍污染了 full training path，哪怕 P3 短程 path audit 通过。

H2 成立标准：

$$
near\_pass_{\text{LQ}}=1
$$

but:

$$
near\_pass_{\text{TPEA-off}}=0.
$$

此时必须回到 attach infrastructure，不能测试 functional carrier。

## H3：checkpoint-level late attach 是比 registered dormant 更干净的 functional carrier 路线

Registered dormant 即使 optimizer-excluded，也可能影响 module registry、parameter ordering、param matching、state dict、memory accounting。  
Late attach 在 base phase 完全不存在 functional channel，理论上不会污染 base。

H3 成立标准：

Late attach inactive checkpoint equivalence pass，且 no-event replay 与 base checkpoint完全一致：

$$
\max_x\|z_{\text{late-off}}-z_{\text{base}}\|_\infty\leq10^{-6}.
$$

## H4：functional carrier 必须在 checkpoint 后测试，而不是用 functional update 救 base

Functional update 不能救 P5 failed base。  
只有在：

```text
LQ base anchor pass
snapshot attach inactive equivalence pass
```

之后，functional event 才能打开。

H4 成立标准：

所有 official functional rows 均引用已通过 base anchor 的 checkpoint hash。没有 hash 的 row 不允许进入 route。

## H5：如果 late attach 能动但 value 不过，blocker 才回到 value alignment

如果 late attach 保住 base、carrier non-silent，但 paired replay仍 control-equivalent，则说明 blocker 是 value statistic / role-wise mechanism，而不是 base preservation。

H5 成立标准：

$$
r_{z,\text{tail}}\geq0.10
$$

and:

$$
r_{\perp,\text{tail}}\geq0.10
$$

but:

$$
AUC(Y_{\text{beat}})<0.70
$$

or paired replay beat rate < gate.

## H6：任何 dataset-specific route 都不得 official pass

允许记录不同 dataset slice 上的失败模式，但 controller 只能使用 signal strata / event features：

```text
CEp99
MarginP10
Curvature
ControlGap
BranchRatio
EffectiveDerivative
RoleSignal
TailMask
Uncertainty
```

不能使用：

```text
dataset_name
seed_id
class-specific handcrafted rule
validation metric
test metric
posthoc replay outcome at commit
```

---

# Part IV. Candidate 设计

## 1. Base candidates

### B0：Historical LQ Reference

来自 v9.2.6 / v9.2.7 的 reference metrics。  
不直接作为模型运行，只用于 comparison table。

### B1：Current LQ Reproduction

当前代码路径下重新运行：

```text
LQ-t2-h256
same hidden
same fan-in output scale
same compact memory mode
same dataset split
same seeds
same MLP-match definition
```

### B2：Current LQ Frozen Checkpoint

从 B1 训练得到的 task-channel checkpoint：

$$
\theta_T^\star.
$$

这是所有 late-attach functional 实验的唯一 official base source。

---

## 2. Attach candidates

### A0：No-Attach LQ Checkpoint

Reference branch:

$$
z(x)=z_{\text{LQ}}(x;\theta_T^\star).
$$

### A1：RegisteredZeroExcluded

复现 v9.2.37 best TPEA1：

```text
functional params registered
zero-init
excluded from optimizer
lambda=0
```

### A2：LateAttachZeroLinearTail

Checkpoint 后 attach:

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T^\star)
+
\lambda(e)a_{ij}q(h)h.
$$

Inactive:

$$
\lambda(e)=0.
$$

### A3：LateAttachControlGapChannel

Functional channel only accepts event if:

$$
LCB(Gain_{\text{Real}})
>
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### A4：LateAttachFamilyValueChannel

Event family:

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

### A5：LateAttachRoleWiseFT7EdgeCarrier

Edge-owned role-wise functional channel:

$$
\phi_{ij}(h)
=
\phi^{LQ}_{ij}(h;\theta_T^\star)
+
\lambda(e)
[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
].
$$

Role weights are signal-based:

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

### A6：SymbolicFunctionalSpecAttach

Functional carrier exists as symbolic spec until event trigger。  
No functional params are registered before event.

```text
base phase:
  no theta_F

event phase:
  instantiate theta_F as edge-owned zero-init
  apply one functional update
  evaluate carrier / value / paired replay
```

---

# Part V. 实验阶段

## P0：v9.2.37 boundary reproduction

### 目标

复现 v9.2.37 terminal boundary，确认不是 measurement artifact。

### 必须记录

```text
route
source_route_v9236
bpfs_failure_mode
bpfs_failure_primary_fraction
tpea_implemented_count
training_path_equivalence_pass
task_init_hash_match
optimizer_state_match
lambda_inactive_leak_detected
tpea_contract_pass
tpea_grad_pass
best_tpea_candidate
tpea_p4_pass
tpea_p5_nearpass
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
route = R11-TPEAAllFailPrimitiveReset
training_path_equivalence / contract / grad = 1 / 1 / 1
P5 near-pass = 0
carrier pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_tpea_gate_ladder.svg
p0_equivalence_vs_p5_failure.svg
```

---

## P1：LQ base anchor and protocol mismatch audit

### 目标

回答：当前 P5 failure 是 LQ base 本身不复现，还是 TPEA attach 污染？

### 设置

```text
candidates:
  HistoricalLQReference
  CurrentLQReproduction
  TPEA1-off
  TPEA3-late-attach-off if implemented

datasets:
  MNIST,Fashion-MNIST,KMNIST

seeds:
  0,1,2 and historical reference seeds if available

functional_update:
  off

lambda:
  0
```

### 必须记录

```text
candidate
dataset
seed
hidden_dim
basis_type
fan_in_output_scale
memory_mode
train_size
eval_size
batch_size
lr
epochs
seed_protocol_hash
data_split_hash
minibatch_order_hash
mlp_match_param_count
kan_param_count
param_ratio
MLP_match_acc
KAN_acc
historical_LQ_acc
delta_vs_mlp
delta_vs_historical_LQ
near_pass
CEp99
margin_p10
ECE
NLL
```

Protocol fields:

```text
p5_gate_definition_hash
mlp_match_definition_hash
candidate_config_hash
runner_hash
data_protocol_hash
```

### 判断标准

LQ anchor pass：

$$
near\_pass\_rate_{\text{CurrentLQ}}\geq0.80,
$$

$$
\Delta Acc_{\text{macro,CurrentLQ}}\geq-0.01,
$$

and:

$$
|\Delta Acc_{\text{CurrentLQ}}-\Delta Acc_{\text{HistoricalLQ}}|\leq0.003.
$$

TPEA-off P5 equivalence pass：

$$
|\Delta Acc_{\text{TPEA-off}}-\Delta Acc_{\text{CurrentLQ}}|\leq0.001.
$$

Protocol mismatch detected if any hash differs without explicit justification.

### 可视化

```text
p1_lq_reanchor_matrix.svg
p1_historical_vs_current_lq.svg
p1_tpea_vs_lq_p5_delta.svg
p1_protocol_hash_diff.svg
```

---

## P2：Snapshot late-attach implementation

### 目标

实现 checkpoint-level late attach。Base phase 不存在或不使用 functional channel；attach 后 inactive 必须等价 checkpoint。

### 必须记录

```text
candidate
attach_mode
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
A2-A6 至少实现 4 个
A3 or A5 至少实现 1 个
A6 optional but preferred
all implemented candidates smoke forward/backward/update pass
strict PureKAN contract pass
```

Contract pass：

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
p2_snapshot_attach_matrix.svg
p2_param_state_hash_stability.svg
```

---

## P3：Checkpoint inactive equivalence audit

### 目标

证明 attach 后 $\lambda=0$ 时不改变 base checkpoint。

### 必须记录

```text
candidate
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

Inactive checkpoint equivalence pass：

$$
\max_x
\|z_{\text{attach-off}}-z_{\text{LQ checkpoint}}\|_\infty
\leq10^{-6}.
$$

Task parameter stability：

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
p3_inactive_logit_diff.svg
p3_task_param_state_diff.svg
p3_lambda_leak_audit.svg
```

---

## P4：No-event replay / base preservation after attach

### 目标

证明 attach 后在 no-event setting 下仍与 base checkpoint continuation 一致。

### 设置

```text
base = CurrentLQ checkpoint
branches:
  LQ checkpoint continue
  attach-off continue
  attach-no-event continue
steps = 1,5,20,80
functional_update = off
events = none
```

### 必须记录

```text
candidate
dataset
seed
steps
branch
train_loss
holdout_loss
val_acc_proxy
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
|CEp99_{\text{attach-no-event}}-CEp99_{\text{base-continue}}|\leq10^{-5}.
$$

$$
|Margin_{\text{attach-no-event}}-Margin_{\text{base-continue}}|\leq10^{-5}.
$$

System gate：

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
p4_system_overhead_no_event.svg
```

---

## P5：Functional carrier actuatability after late attach

### 目标

只在 P1-P4 通过的 candidate 上测试 event-time functional channel 是否能动。  
P5 不是 paired replay success。

### 设置

```text
base = qualified LQ checkpoint
candidates = P4 survivors only
events = signal strata S1-S8
horizons = 1,5,20,80,240
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
```

### 必须记录

```text
candidate
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
```

---

## P6：Value observability audit

### 目标

判断 late-attach functional score 是否预测 grounded Real-vs-control value。

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
p6_auc_precision_coverage_by_attach.svg
p6_score_components_correlation.svg
p6_accepted_strata_distribution.svg
```

---

## P7：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 late-attach controller 隐式 dataset tuning。

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

## P8：Official snapshot late-attach paired replay

### 目标

只有 P1-P7 survivor 可以进入 official paired replay。

### 设置

```text
base = qualified LQ checkpoint
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
candidate = best late-attach survivor
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

## P11：AdamW-only LQ base full-pass repair

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

# Part VI. Required artifacts

```text
run_manifest.json
contract_audit_v9238.csv
p0_v9237_boundary_reproduction.csv
p1_lq_base_anchor_protocol_audit.csv
p2_snapshot_late_attach_implementation.csv
p3_checkpoint_inactive_equivalence.csv
p4_no_event_replay_base_preservation.csv
p5_functional_carrier_actuatability.csv
p6_value_observability_audit.csv
p7_leave_dataset_and_stratum_out_validation.csv
p8_official_snapshot_late_attach_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_10seed_functional_validation.csv
p11_adamw_only_lq_fullpass_repair.csv
p12_robustness_external_ready.csv
lq_anchor_trace_v9238.csv
protocol_hash_diff_v9238.csv
snapshot_attach_trace_v9238.csv
checkpoint_equivalence_trace_v9238.csv
functional_carrier_trace_v9238.csv
value_score_trace_v9238.csv
leave_dataset_out_trace_v9238.csv
paired_replay_branch_trace_v9238.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9237_boundary_unstable
F3_dataset_tuning_detected
F4_lq_base_anchor_fail
F5_historical_current_lq_drift
F6_mlp_match_definition_drift
F7_candidate_config_hash_mismatch
F8_data_split_or_seed_protocol_drift
F9_tpea_off_not_p5_equivalent_to_lq
F10_snapshot_late_attach_not_implemented
F11_checkpoint_inactive_equivalence_fail
F12_no_event_replay_preservation_fail
F13_functional_carrier_silent
F14_value_observability_fail
F15_value_score_system_too_expensive
F16_leave_dataset_out_fail
F17_leave_stratum_out_fail
F18_paired_replay_control_equivalent
F19_shuffle_control_pass
F20_functional_lr_equivalent
F21_short_run_task_drop
F22_full_run_no_macro_hard_stratum_gain
F23_adamw_fullpass_fail
F24_strong_baseline_explains_gain
F25_robustness_fail
F26_external_not_ready
F27_fake_or_proxy_violation
F28_artifact_missing
```

---

# Part VII. Route decision

## Route cases

```text
R1-LQBaseAnchorBroken:
  current LQ reference cannot reproduce historical near-pass.

R2-ProtocolMismatchFound:
  LQ failure attributed to candidate config / MLP-match / seed / data split / gate drift.

R3-LQBaseReanchored:
  current LQ reference passes P4/P5 and matches historical tolerance.

R4-TPEAOffEquivalentToLQ:
  TPEA-off matches current LQ in P5.

R5-SnapshotLateAttachImplemented:
  checkpoint-level late attach implemented under strict PureKAN contract.

R6-SnapshotAttachInactiveEquivalent:
  attached model inactive equals base checkpoint.

R7-NoEventReplayPreserved:
  no-event replay remains equivalent to base continuation.

R8-FunctionalCarrierActive:
  carrier produces non-silent, non-AdamW movement.

R9-ValueObservabilityPass:
  value score predicts grounded Real-vs-control value.

R10-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R11-SnapshotLateAttachPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-LQBaseFailNoFunctional:
  base anchor fails; functional route not allowed.

R13-BasePreservedButFunctionalSilent:
  base is preserved, but carrier cannot move output.

R14-BasePreservedButValueUnobservable:
  carrier moves output, but value score fails.

R15-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R16-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R17-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R18-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

## route_decision.json 必须记录

```text
route
v9237_boundary_pass
dataset_tuning_detected
lq_base_anchor_pass
historical_current_lq_delta
protocol_mismatch_detected
current_lq_nearpass
tpea_off_p5_equivalence_pass
snapshot_attach_implemented_count
best_late_attach_candidate
checkpoint_inactive_equivalence_pass
no_event_replay_preservation_pass
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
success_v9238_strict_purekan_functional
success_v9238_full_functional
success_v9238_external_ready
```

---

# Part VIII. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.37 boundary。

Step 2:
  P1 做 LQ base anchor / protocol mismatch audit。
  不要先实现新 functional carrier；先确认当前 LQ reference 是否还成立。

Step 3:
  如果 LQ anchor fail：
    停止 functional route，进入 LQ protocol repair / base re-anchor。

Step 4:
  如果 LQ anchor pass：
    对 TPEA-off 做 P5 equivalence。
    若 TPEA-off 不等价，修 attach infrastructure。

Step 5:
  P2 实现 snapshot late attach。
  Base phase 不注册或不使用 functional params。

Step 6:
  P3 做 checkpoint inactive equivalence。

Step 7:
  P4 做 no-event replay preservation。

Step 8:
  P5 测 event-time functional carrier actuatability。

Step 9:
  P6 测 value observability。

Step 10:
  P7 做 leave-dataset-out / leave-stratum-out。

Step 11:
  P8 official paired replay。

Step 12:
  P9 short-run validation。

Step 13:
  P10 full 10-seed validation。

Step 14:
  P11 并行 AdamW-only LQ full-pass repair。

Step 15:
  P12 robustness / strong baseline / external-ready。
```

---

# Part IX. 停止条件

## Minimum diagnostic success

```text
v9.2.37 boundary reproduced
LQ base anchor status resolved
protocol mismatch either found or ruled out
snapshot late attach implemented if LQ anchor pass
no fake/proxy/offload/loss/teacher violation
```

## Base-ready attach success

```text
LQ base anchor pass
+
TPEA-off or snapshot attach inactive equivalence pass
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

## External-ready success

```text
Full functional success
+
robustness pass
+
strong baseline challenge pass
```

## Failure stop

```text
1. v9.2.37 boundary cannot be reproduced；
2. current LQ base cannot reproduce historical near-pass and protocol mismatch cannot be resolved；
3. TPEA-off differs from LQ despite same current protocol；
4. snapshot late attach cannot be implemented as strict PureKAN；
5. inactive checkpoint equivalence fails；
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

# Part X. 最终解释规则

## Case A：Current LQ base fails

必须声明：

```text
The functional route is blocked because the LQ base anchor is no longer reproduced under the current protocol.
```

下一步先修 LQ base/protocol，不允许 functional update 继续推进。

## Case B：Current LQ passes, TPEA-off fails

必须声明：

```text
The attach infrastructure still contaminates the base path despite short training-path audit.
```

下一步修 TPEA / attach infrastructure。

## Case C：Snapshot late attach preserves base

可以声明：

```text
The base-preservation problem is resolved at checkpoint level.
```

但不能声明 functional success，除非 carrier / observability / paired replay 也通过。

## Case D：Base preserved but carrier silent

必须声明：

```text
The base is preserved, but the functional subspace lacks event-time actuatability.
```

下一步修 carrier，而不是 dataset patch。

## Case E：Carrier active but value unobservable

必须声明：

```text
Functional movement exists but is not control-resistant value.
```

下一步回到 value statistic / role-wise FT7 mechanism。

## Case F：Paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XI. 最终建议

v9.2.38 的一句话策略是：

$$
\boxed{
\text{先重新锚定 LQ base；再用 checkpoint-level late attach 测 functional，而不是继续在未锚定 base 上调 carrier。}
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
1. 当前 LQ-t2-h256 是否还能在同一协议下复现 P4/P5 near-pass？
2. TPEA-off 的 P5 failure 是 LQ base gap，还是 attach infrastructure 污染？
3. checkpoint-level late attach 能否做到 inactive 等价和 no-event replay 等价？
4. base 重新锚定后，functional carrier 是否能产生 non-silent, non-AdamW movement？
5. control-gap / role-wise value score 是否能预测 grounded Real-vs-control value？
6. 这个规则能否 leave-dataset-out 泛化？
7. RealFunctional 能否在 official paired replay 中超过 AdamWParallel / best LR？
```
