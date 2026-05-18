# DG-KAN v9.2.35 Value-Aligned Observable Primitive：从“可实现但不可预测”到“价值因子化 functional primitive”的完整实验计划

> 本计划基于 v9.2.34 `Observable Primitive First` 的真实复盘结果制定。  
> v9.2.34 的 terminal route 是：
>
> ```text
> route = R6-ObservablePrimitiveEffectUnpredictable
> base_candidate = LQ-t2-h256
> success_v9234_observable_primitive = False
> success_v9234_strict_purekan_functional = False
> success_v9234_external_ready = False
> ```
>
> v9.2.34 的关键进展是：OP1-OP6 不再只是 `not_implemented`，已经进入真实 implementation smoke；其中一部分 candidate 能进入 contract / grad / P4 / P5 qualification。  
> v9.2.34 的关键失败是：即使 best OP4 `ObservableOrthogonalTailChannel` 已经是当前 best base-qualified OP，它仍然不能预测 grounded event value：
>
> ```text
> best OP = OP4-ObservableOrthogonalTailChannel
> best observability corr = 0.099935
> best observability AUC = 0.415519
> accepted precision = 0.666667
> accepted coverage = 0.111111
> bad-event rate = 0.000000
> observability pass = 0
> ```
>
> 因此 v9.2.35 不再继续做 Fashion / KMNIST / MNIST 的数据集特化调参，也不再继续单纯调 OP score threshold。  
> v9.2.35 的核心任务是：
>
> $$
> \boxed{
> \text{从“movement-observable primitive”转向“value-aligned observable primitive”。}
> }
> $$
>
> 换句话说，v9.2.34 已经说明：**把 primitive 做成可实现、可训练、局部/正交/可观测 movement，不等于它能预测或产生 control-resistant value。**  
> 下一步必须让 functional primitive 的结构本身暴露一个和 Real-vs-control value 对齐的 sufficient statistic，而不是事后希望 movement score 与 value 相关。

---

## 0. 本轮路线约束

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

数据集只能作为诊断切片，不能作为 official controller 的条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice failures
  report leave-dataset-out generalization
  report which signal strata dominate each dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
```

---

## 1. v9.2.34 的独立判断

### 1.1 没有达到目标

v9.2.34 没有达到 observable primitive success，也没有达到 strict PureKAN functional success。其 route 是：

```text
R6-ObservablePrimitiveEffectUnpredictable
```

关键结果是：

```text
implemented OP count = 6
contract/grad/P4 eligible count = 5
P4 pass count = 2
P5 near-pass count = 1
best OP = OP4-ObservableOrthogonalTailChannel
best observability corr = 0.099935
best observability AUC = 0.415519
accepted precision = 0.666667
accepted coverage = 0.111111
bad-event rate = 0.000000
```

这说明 OP implementation gap 被推进了，但 observability gate 没有闭合。  
尤其 AUC 低于 $0.5$，说明 OP4 的 score 不只是“弱”，而是存在方向错配或 value label 对齐失败。accepted coverage 已经在合理区间内，bad-event 为 0，但 precision 只有 $0.666667$，低于 $0.75$ gate。这意味着：OP4 产生的 accepted events 大多不伤任务，但也不能稳定击败 AdamWParallel / best LR。

### 1.2 这不是“当前基函数全部失败”

`LQ-t2-h256` 仍然是当前 strict FC-PureKAN base scaffold。它承担 task channel / base qualification / P4-P5 的角色。  
v9.2.34 的失败点在 functional channel / observable primitive，不在 LQ/T2 base 本身。

因此当前判断是：

$$
\boxed{
\text{保留 LQ/T2 base；重做 functional primitive。}
}
$$

### 1.3 不能继续小修小补

v9.2.34 后，继续做下面这些事情意义不大：

```text
1. 继续调 OP4 threshold；
2. 继续只比较 movement ratio；
3. 继续在 OP0/N2a/N2c/N3c 上换 probe；
4. 按 Fashion / KMNIST 写 dataset-specific controller；
5. 直接 short-run / full-run；
6. 把 bad-event=0 包装成 success。
```

真正的问题是：

$$
\boxed{
\text{当前 OP family 的 observable score 没有和 control-relative value 因子化对齐。}
}
$$

---

## 2. v9.2.35 总体目标

v9.2.35 的总体目标是：

$$
\boxed{
\text{设计并验证 value-aligned observable primitive，使 commit 前 score 能预测 RealFunctional 是否超过强 controls。}
}
$$

这个目标分为五层。

### 2.1 Failure attribution success

先解释 v9.2.34 为什么出现：

```text
coverage acceptable
bad-event = 0
precision insufficient
AUC < 0.5
```

必须把 OP failure 归因到以下机制之一：

```text
F1-score_sign_mismatch:
  score 方向与 grounded value 相反或部分相反。

F2-orthogonality_not_value:
  非 AdamW 正交分量存在，但它不对应 CE-tail / margin-tail / curvature value。

F3-tail_mask_not_causal:
  tail mask 选中了 high-risk 样本，但 update 对这些样本的 value 不稳定。

F4-control_gap_unmodeled:
  Real 有 non-harmful effect，但 AdamWParallel / best LR 更强。

F5-primitive_too_global:
  movement 分散到非 tail samples，导致 value diluted。

F6-primitive_too_local:
  local correction 不能累计到 metric-level gain。

F7-value_label_high_variance:
  grounded value 可靠性不足以支持 single-event score，但 family-level value 可预测。

F8-base_channel_conflict:
  functional channel 干扰 task channel 或被 task channel 吸收。
```

### 2.2 Value-aligned primitive success

新的 primitive 不以 movement 大为目标，而以 pre-commit value statistic 为目标：

$$
S_{\text{VAOP}}
\approx
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

成功标准：

$$
AUC(Y_{\text{beat}})\geq0.70,
$$

或：

$$
Corr(S_{\text{VAOP}},V_{\text{grounded}})\geq0.35.
$$

并且 accept / abstain gate：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 2.3 Base/system success

value-aligned primitive 不能破坏 strict PureKAN base：

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

### 2.4 Dataset-agnostic success

Official controller 不使用 dataset name。必须做 leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

通过标准：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 split 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

### 2.5 Local functional causality success

只有 primitive observability + LDO 通过后，才能进入 official paired replay。paired replay 成功标准：

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

---

## 3. 核心假设

### H1：v9.2.34 的失败不是 implementation failure，而是 value alignment failure

OP1-OP6 已实现，且有 candidate 能进 P4 / P5；但 best OP4 AUC 只有 $0.415519$，corr 只有 $0.099935$。  
这说明当前 primitive 已经从 “没有实现” 推进到 “实现了但不预测 value”。

H1 成立标准：

P1 failure attribution 中，OP4 / OP family 的 score 与 grounded value 主要落入：

```text
F2-orthogonality_not_value
F4-control_gap_unmodeled
F5/F6 locality mismatch
```

而不是：

```text
contract fail
grad fail
P4 fail
P5 fail
data artifact
```

### H2：正交 tail movement 不是 sufficient statistic

OP4 的思想是 orthogonal tail channel，但 AUC 低于随机。  
H2 认为：

$$
r_{\perp,\text{tail}}
$$

不能单独作为 functional value 的 proxy。真正需要的是：

$$
\langle \Delta z_{\text{tail}}, -\nabla_z L_{\text{tail}}\rangle
-
S_{\text{control}}
-
Penalty_{\text{risk}}.
$$

H2 成立标准：

若 $r_{\perp,\text{tail}}$ 与 $V_{\text{grounded}}$ 的 corr 明显低于 value-aligned score corr，则 H2 成立。

### H3：value-aligned primitive 应该显式使用 task-gradient-in-output-space，但不能改 loss

Functional update 可以使用标准 CE 的输出梯度作为 update rule 的信号：

$$
g_z=\nabla_z CE.
$$

这不是改 loss，因为 task objective 仍是 CE。  
Value-aligned statistic 可以是：

$$
S_{\text{align}}
=
\langle
\Delta z_{\text{func,tail}},
-g_{z,\text{tail}}
\rangle
-
\max(S_{\text{AdamWParallel}},S_{\text{bestLR}})
-
\lambda R_{\text{task-risk}}.
$$

H3 成立标准：

VAOP primitive 的 $S_{\text{align}}$ 达到 AUC / corr gate，并且 no teacher/no loss/no dataset route audit 通过。

### H4：role-wise FT7 mechanism 需要以 strict edge-owned primitive 方式重引入

v8-FT7 的 success 不是普通 target patch；它有 role-wise / branch-ratio / curvature 机制。当前 OP family 没有真正复现 FT7 role-wise functional interface。  
H4 认为，应设计 strict edge-owned dual-role primitive：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
\phi^{func}_{ij}(h),
$$

其中 task channel 继续由 LQ/T2 稳定训练，functional channel 只在 value-aligned event 中更新。

H4 成立标准：

VAOP-RoleWise primitive 在 P4/P5 不破坏 base，同时 observability / paired replay 过 gate。

### H5：如果 value-aligned primitive 也失败，应回到更深 edge-function interface，而不是 dataset patch

如果 VAOP1-VAOP6 仍然无法同时满足 P4/P5/observability，说明当前 strict FC-PureKAN functional family 缺少可用 interface。下一步应回到 role-wise edge function / basis factory / functional channel parameterization，而不是 Fashion/KMNIST-specific patch。

---

## 4. Candidate 设计

## 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  strict FC-PureKAN AdamW-only base。

OP0-current-reference:
  current N2a/N2c/N3c family reference。

OP4-current-best:
  ObservableOrthogonalTailChannel from v9.2.34。

V8-FT7:
  mechanism source only，不作为 official strict candidate。
```

## 4.2 Value-aligned observable primitives

### VAOP1：TailGradientAlignedLinearChannel

目标：最小化设计复杂度，直接测试 output-space value alignment 是否是缺失因素。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
a_{ij}\cdot q(h)\cdot h.
$$

其中：

$$
q(h)=\sigma(\gamma(|h|-\tau)).
$$

Functional update 不直接最大化 movement，而是选择能提高 tail CE/margin value 的 $\Delta a$。  
Pre-commit score：

$$
S_{\text{VAOP1}}
=
\langle
\Delta z_{\text{func,tail}},
-g_{z,\text{tail}}
\rangle
-
\max(S_{\text{AdamWParallel}},S_{\text{bestLR}})
-
\lambda\|\Delta z_{\text{non-tail}}\|^2.
$$

### VAOP2：MarginJacobianPiecewiseChannel

目标：针对 margin-tail 的局部函数变化，而不是全局 CE movement。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
\sum_{k=1}^{K}
a_{ij,k}
\operatorname{clip}(h-\tau_k,0,\Delta).
$$

默认：

```text
K = 2,4
shared knots
bounded slope
no B-spline recursion
```

Pre-commit score：

$$
S_{\text{VAOP2}}
=
\langle
\Delta m_{\text{tail}},
\mathbf{1}_{m<q_{10}}
\rangle
-
S_{\text{control}}
-
R_{\text{wrong-confidence}}.
$$

其中 $m$ 是 true-class logit margin。

### VAOP3：ControlGapBoundedSharedRBFChannel

目标：用 shared RBF 的 local support 形成可解释 basis-energy statistic。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
\sum_{k=1}^{K}
a_{ij,k}
\exp
\left(
-\frac{(h-\mu_k)^2}{2\sigma^2}
\right).
$$

Pre-commit score：

$$
S_{\text{VAOP3}}
=
\sum_k
E_{\text{tail},k}
\Delta a_k
-
UB(S_{\text{control}})
-
\lambda EntropyPenalty.
$$

要求 basis energy normalization：

$$
\sum_k E_{\text{tail},k}=1.
$$

### VAOP4：RoleWiseFT7EdgeChannel

目标：以 strict PureKAN edge-owned 形式重引入 FT7 的 role-wise functional interface。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
\phi^{func,stack}_{ij}(h)
+
\phi^{func,head}_{ij}(h).
$$

Functional role update：

$$
\Delta\theta_{\text{func}}
=
\alpha_s\Delta\theta_{\text{stack}}
+
\alpha_h\Delta\theta_{\text{head}}.
$$

Role weights 不按 dataset 调，而由 signal strata 决定：

$$
\alpha_r
=
f(
CEp99,
MarginP10,
Curvature,
BranchRatio,
EffectiveDerivative
).
$$

Score：

$$
S_{\text{VAOP4}}
=
S_{\text{tail-value}}
+
\eta S_{\text{curvature}}
-
S_{\text{control}}
-
R_{\text{task-risk}}.
$$

### VAOP5：ConservativeControlGapLowerBoundChannel

目标：只接受有 control-gap lower bound 的事件，宁可 coverage 低，也先证明 precision。

定义：

$$
LCB_{\text{Real}}=\mu_{\text{Real}}-\kappa\sigma_{\text{Real}},
$$

$$
UCB_{\text{Control}}=\max_c(\mu_c+\kappa\sigma_c).
$$

接受条件：

$$
LCB_{\text{Real}}>UCB_{\text{Control}}.
$$

成功不能只看 precision，还必须 coverage：

$$
Coverage\geq0.03.
$$

### VAOP6：FamilyValueChannel

目标：若 single-event value 太噪，则按 signal family 学习 event-family value，但仍不按 dataset 分支。

Family 定义：

$$
family(e)
=
(stratum(e), primitive(e), horizon(e), risk\_bucket(e)).
$$

接受条件：

$$
\mathbb{E}[V|family]>0
$$

且 family reliability：

$$
Reliability(family)\geq0.30.
$$

---

## 5. 实验阶段

## P0：v9.2.34 boundary reproduction

### 目标

复现 v9.2.34 boundary，确认当前失败不是 measurement artifact。

### 必须记录

```text
route
source_route_v9233
implemented_op_count
contract_grad_p4_eligible_count
p4_pass_count
p5_nearpass_count
best_op
best_base_qualified_op
best_base_macro_delta
best_base_step_q90
best_observability_corr
best_observability_auc
accepted_precision
accepted_coverage
accepted_bad_event_rate
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R6-ObservablePrimitiveEffectUnpredictable
implemented_op_count = 6
observability pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_op_gate_ladder.svg
p0_observability_failure_recap.svg
```

---

## P1：OP failure attribution

### 目标

解释 OP1-OP6 为什么实现后仍不能预测 grounded value。  
P1 不实现新 primitive，只分析 v9.2.34 的 OP traces。

### 必须记录

```text
primitive
controller
event_id
signal_stratum
dataset_slice
horizon
obs_score
grounded_value
Y_beat
score_rank
value_rank
score_sign
movement_ratio
tail_movement_ratio
orthogonal_tail_ratio
control_gap
CEp99_delta
margin_delta
curvature_delta
task_risk
failure_mode
```

### 判断标准

P1 pass：

```text
>= 90% failed events assigned to primary failure mode
failure attribution does not use dataset_name
OP4 failure specifically classified
AUC<0.5 reason explained as sign / control / locality / variance issue
```

定量分析：

$$
Corr(r_{\perp,\text{tail}},V_{\text{grounded}})
$$

$$
Corr(S_{\text{obs}},V_{\text{grounded}})
$$

$$
Corr(S_{\text{control-gap}},V_{\text{grounded}})
$$

必须分别记录，不能只报 best score。

### 可视化

```text
p1_op_score_vs_value.svg
p1_failure_mode_sankey.svg
p1_score_sign_mismatch.svg
p1_control_gap_dominance.svg
p1_movement_vs_value.svg
```

---

## P2：Value-aligned primitive implementation

### 目标

实现 VAOP1-VAOP6。  
每个 primitive 必须是 strict edge-owned PureKAN，不允许 external residual / ordinary MLP path。

### 必须记录

```text
primitive
basis_formula
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
functional_channel_type
task_channel_type
basis_count
shared_basis
bounded_derivative
analytic_value_stat
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
VAOP1-VAOP6 至少实现 4 个；
每个实现 candidate 能 forward/backward/update smoke；
每个 primitive 的 analytic_value_stat 必须可在 commit 前计算；
未实现项必须写明技术原因。
```

### 可视化

```text
p2_vaop_implementation_matrix.svg
p2_value_stat_by_primitive.svg
```

---

## P3：Contract / gradcheck / interaction audit

### 目标

证明 VAOP primitives 仍然是 strict PureKAN，并且梯度正确。

### 必须记录

```text
primitive
contract_pass
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
non_edge_owned_param_count
manual_forward
manual_backward
manual_update
uses_loss_backward
GradRelErrMax
GradCosMin
pairwise_R2
local_bump_R2
basis_condition_number
task_channel_entropy
functional_channel_entropy
dominant_basis_fraction
```

### 判断标准

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

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

Interaction pass：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

如果 primitive 是 deliberately local，则必须同时报告 local-bump R2：

$$
R^2_{\text{local-bump}}\geq0.95.
$$

### 可视化

```text
p3_contract_grad_matrix.svg
p3_interaction_and_local_fit.svg
p3_basis_conditioning.svg
```

---

## P4：P4/P5 base qualification

### 目标

VAOP 不能破坏 base trainability。  
functional_update 关闭，只测 AdamW-only base。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
baseline = MLP-match
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
kernel_count
extra_probe_cost
```

Task：

```text
dataset
seed
KAN_acc
MLP_match_acc
delta_vs_mlp
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

### 可视化

```text
p4_system_pareto.svg
p4_task_nearpass_matrix.svg
p4_base_gap_by_vaop.svg
```

---

## P5：Value observability audit

### 目标

判断 VAOP 是否真正把 causal value 暴露为 commit-time score。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240
events = signal strata S1-S8
controllers = VAOP score variants
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
primitive
controller
event_id
signal_stratum
horizon
value_score
tail_value_score
control_gap_score
role_score
grounded_value
Y_beat
corr
auc
precision
coverage
bad_event_rate
accepted_event_count
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
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
p5_value_score_vs_grounded_value.svg
p5_auc_precision_coverage_by_vaop.svg
p5_value_system_pareto.svg
p5_accepted_strata_distribution.svg
```

---

## P6：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 VAOP controller 隐式 dataset tuning。

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
primitive
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

至少两个 LDO splits 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

至少 $70\%$ held-out strata task-safe 且 CE tail 不劣化：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p6_leave_dataset_out_matrix.svg
p6_leave_stratum_out_matrix.svg
p6_hidden_dataset_tuning_audit.svg
```

---

## P7：Official value-aligned paired replay

### 目标

只允许 P2-P6 survivor 进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
primitive = best VAOP survivor
controller = best legal signal-based controller
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, PrimitiveScoreShuffled, TailMaskShuffled, SignalScoreShuffled, ValueScoreShuffled
```

### 必须记录

```text
primitive
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
PrimitiveScoreShuffled = fail
TailMaskShuffled = fail
SignalScoreShuffled = fail
ValueScoreShuffled = fail
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
p7_official_paired_replay_pareto.svg
p7_macro_beat_rate.svg
p7_signal_stratum_win_matrix.svg
p7_shuffle_control_matrix.svg
p7_system_gate_distribution.svg
```

---

## P8：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P7 survivor 才能进入。

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
p8_short_run_task_mechanism_pareto.svg
p8_short_run_controls.svg
p8_event_timeline.svg
p8_ce_tail_margin_panel.svg
```

---

## P9：Full 10-seed functional validation

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

### 可视化

```text
p9_macro_delta_vs_controls.svg
p9_hard_stratum_repair_matrix.svg
p9_seedwise_win_matrix.svg
p9_task_geometry_pareto.svg
p9_ce_tail_margin_panel.svg
p9_functional_channel_usage_trace.svg
```

---

## P10：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。P10 并行执行，不使用 functional update，不使用 teacher/loss/sampler/class weight。

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
p10_adamw_fullpass_gap.svg
p10_hard_stratum_miss_rows.svg
p10_ce_tail_margin.svg
p10_basis_entropy_vs_gap.svg
```

---

## P11：Robustness and external-ready gate

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
p11_noise_robustness_curve.svg
p11_strong_baseline_pareto.svg
p11_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9235.csv
p0_v9234_boundary_reproduction.csv
p1_op_failure_attribution.csv
p2_value_aligned_primitive_implementation.csv
p3_contract_grad_interaction_audit.csv
p4_vaop_p4_p5_base_qualification.csv
p5_value_observability_audit.csv
p6_leave_dataset_and_stratum_out_validation.csv
p7_official_value_aligned_paired_replay.csv
p8_short_run_functional_validation.csv
p9_full_10seed_functional_validation.csv
p10_adamw_only_fullpass_repair.csv
p11_robustness_external_ready.csv
vaop_primitive_trace_v9235.csv
value_score_trace_v9235.csv
op_failure_trace_v9235.csv
leave_dataset_out_trace_v9235.csv
paired_replay_branch_trace_v9235.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9234_boundary_unstable
F3_dataset_tuning_detected
F4_op_failure_unattributed
F5_vaop_not_implemented
F6_vaop_contract_fail
F7_vaop_grad_fail
F8_vaop_p4_fail
F9_vaop_p5_fail
F10_vaop_observability_fail
F11_vaop_system_too_expensive
F12_leave_dataset_out_fail
F13_leave_stratum_out_fail
F14_paired_replay_control_equivalent
F15_shuffle_control_pass
F16_functional_lr_equivalent
F17_short_run_task_drop
F18_full_run_no_macro_hard_stratum_gain
F19_adamw_fullpass_fail
F20_strong_baseline_explains_gain
F21_robustness_fail
F22_external_not_ready
F23_fake_or_proxy_violation
F24_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-OPFailureAttributed:
  v9.2.34 OP failure has mechanism attribution.

R2-ValueAlignedPrimitiveImplemented:
  at least four VAOP primitives implemented and audited.

R3-VAOPContractPass:
  at least one VAOP passes contract / grad / interaction.

R4-VAOPBaseQualified:
  at least one VAOP passes P4 and P5 near-pass.

R5-VAOPObservabilityPass:
  VAOP exposes commit-time value signal.

R6-LeaveDatasetOutVAOPPass:
  VAOP controller generalizes without dataset-specific tuning.

R7-ValueAlignedPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R8-ObservableButTooExpensive:
  observability passes but system envelope fails.

R9-VAOPAllFailPrimitiveReset:
  VAOP1-VAOP6 fail implementation / contract / observability.

R10-AbstentionOnlyDiagnostic:
  precision high but coverage too low for training advantage.

R11-ControlDominatedSignal:
  Real has effect but optimizer controls dominate.

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
v9234_boundary_pass
dataset_tuning_detected
op_failure_mode
vaop_implemented_count
best_vaop_candidate
vaop_contract_pass
vaop_grad_pass
vaop_p4_pass
vaop_p5_nearpass
vaop_observability_pass
vaop_system_pass
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
success_v9235_strict_purekan_functional
success_v9235_full_functional
success_v9235_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.34 boundary。

Step 2:
  P1 做 OP failure attribution。
  先解释 OP4 为什么 coverage 合理、bad-event 为 0，但 AUC / precision 不过。

Step 3:
  P2 实现 VAOP1-VAOP6。
  不允许继续只调 OP4 threshold。

Step 4:
  P3 做 contract / gradcheck / interaction audit。
  不是 strict PureKAN 的 candidate 直接剔除。

Step 5:
  P4 做 P4/P5 base qualification。
  不能用 functional update 救 base。

Step 6:
  P5 做 value observability audit。
  重点看 value-aligned score，而不是 movement 大小。

Step 7:
  P6 做 leave-dataset-out / leave-stratum-out。
  防止隐式 dataset tuning。

Step 8:
  P7 official paired replay。
  只有 VAOP survivor 才能进入。

Step 9:
  P8 short-run validation。

Step 10:
  P9 full 10-seed validation。

Step 11:
  P10 并行 AdamW-only full-pass repair。

Step 12:
  P11 robustness / strong baseline / external-ready。
```

---

## 9. 停止条件

### Minimum diagnostic success

```text
v9.2.34 boundary reproduced
OP failure attributed
VAOP1-VAOP6 at least 4 implemented
at least one VAOP passes contract / grad
no fake/proxy/offload/loss/teacher violation
```

### Value-aligned primitive success

```text
Minimum diagnostic success
+
at least one VAOP passes P4/P5 near-pass
+
observability pass
+
leave-dataset-out pass
```

### Local functional success

```text
Value-aligned primitive success
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
1. v9.2.34 boundary cannot be reproduced；
2. OP failure cannot be attributed；
3. VAOP1-VAOP6 cannot be implemented as strict PureKAN；
4. all VAOP candidates fail gradcheck；
5. all VAOP candidates fail P4/P5；
6. all VAOP candidates fail observability；
7. VAOP observability passes but system envelope fails；
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

### Case A：VAOP observability passes

可以声明：

```text
v9.2.34 failed because observable primitives were movement-observable but not value-aligned; VAOP now exposes commit-time functional value.
```

但不能声明 full functional success，除非 P7-P9 也通过。

### Case B：VAOP passes observability but fails system

必须声明：

```text
functional value is observable, but current primitive is not kernel-native enough.
```

下一步转 system kernelization，不进入 paired replay。

### Case C：VAOP passes P4/P5 but fails observability

必须声明：

```text
the primitive is trainable and efficient, but still does not expose control-resistant functional value.
```

下一步回到 primitive/interface design，而不是 target patch。

### Case D：leave-dataset-out fails

必须声明：

```text
controller is implicitly dataset-specific; official success is not allowed.
```

不能用 dataset tuning 写成功。

### Case E：all VAOPs fail

必须声明：

```text
current value-aligned primitive family cannot carry dataset-agnostic functional update.
```

下一步应回到 deeper edge-function interface design，尤其重新抽取 v8-FT7 的 role-wise mechanism，而不是继续 Fashion/KMNIST patch。

---

## 11. 最终建议

v9.2.35 的一句话策略是：

$$
\boxed{
\text{不要再追求“可观测 movement”；改为设计“value-aligned observable primitive”。}
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
1. OP4 为什么 AUC < 0.5？
2. orthogonal tail movement 为什么不等于 grounded event value？
3. 什么样的 edge-owned functional channel 能直接暴露 Real-vs-control value？
4. value-aligned score 是否能在不使用 dataset name 的情况下泛化？
5. VAOP 是否仍然 P4/P5 合格？
6. RealFunctional 是否能在 official paired replay 中超过 AdamWParallel / best LR？
7. 如果 VAOP 也不行，是否要重新从 v8-FT7 role-wise mechanism 设计 strict PureKAN primitive？
```
