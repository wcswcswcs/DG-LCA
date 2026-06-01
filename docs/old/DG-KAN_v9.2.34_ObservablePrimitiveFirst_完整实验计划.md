# DG-KAN v9.2.34 Observable Primitive First：从 Legal Probe 失败到可观测 Functional Primitive 的完整实验计划

> 本计划基于 v9.2.33 `Legal Train-Stream Probe 与 Observable Primitive` 的真实复盘制定。  
> v9.2.33 的 terminal route 是：
>
> ```text
> route = R14-ReturnToInterfacePrimitiveDesign
> base_candidate = LQ-t2-h256
> success_v9233_strict_purekan_functional = False
> success_v9233_full_functional = False
> success_v9233_external_ready = False
> ```
>
> 本轮最关键的事实是：**source-logged CP5 的信号没有成功合法化为 train-stream online probe；真实 legal probe 不但预测不过，而且系统开销极高；同时 OP1-OP6 observable primitives 仍未实现。**
>
> 因此 v9.2.34 不能继续做 Fashion / KMNIST / MNIST 方向的特化调参，也不能继续把 source-logged CP5 当作可部署 controller。v9.2.34 的主线必须从 “probe 继续 patch” 转向：
>
> $$
> \boxed{
> \text{实现并验证真正的 observable functional primitive。}
> }
> $$
>
> 本计划继续遵守：
>
> ```text
> no teacher
> no self-teacher
> no distillation
> no loss modification
> no label smoothing
> no focal / margin / calibration loss
> no sampler / class weight
> no CPU offload
> no fake / proxy rows
> KAN path 不使用 PyTorch loss.backward graph
> official controller 不使用 dataset_name 分支
> PureKANConv / PureKANFormer 继续 deferred
> ```
>
> Functional update 仍然是 update rule，不是 loss：
>
> $$
> \theta_{t+1}
> =
> \theta_t
> +
> \Delta\theta_{\text{AdamW}}
> +
> \Delta\theta_{\text{functional}}.
> $$
>
> 任务损失保持标准 CE：
>
> $$
> L_{\text{task}}=CE(y,p_\theta(x)).
> $$

---

## 0. 当前实验结果的独立判断

### 0.1 v9.2.33 没有达到目标

v9.2.33 的真实 route 是：

```text
R14-ReturnToInterfacePrimitiveDesign
```

核心数值是：

```text
P1:
  CP5 classification = CP5LegalizableViaTrainStreamProbe
  CP5 best feature = horizon_agreement_score
  CP5 best feature AUC = 0.856125

P2:
  best legal probe = LP4-ControlContrastiveVirtualProbe
  legal probe corr = 0.182689
  legal probe AUC = 0.468670
  legal probe precision = 0.238095
  legal probe coverage = 0.043210
  legal probe bad-event = 0.000000
  legal probe step q90 = 23.828453
  legal probe memory ratio = 1.660319
  legal probe predictive pass = 0
  legal probe system pass = 0

P5:
  best OP = OP0-current-reference
  best OP corr = 0.134571
  OP1-OP6 not_implemented count = 6

No-fake:
  rows_checked = 3429
  fake/proxy/cpu_offload = 0
```

所以当前不能声明：

```text
strict PureKAN functional success
legal probe-to-commit success
observable primitive success
paired replay success
short-run success
full-run success
external-ready
```

### 0.2 v9.2.33 的真实进展

v9.2.33 最重要的贡献是把两个东西分开了：

```text
source-logged diagnostic signal
!=
legal commit-time train-stream signal
```

v9.2.32 中 CP5 的 AUC 和 precision 看起来很好，但 legality 不过。v9.2.33 重做 legal train-stream probe 后，best legal probe 反而是 LP4，且：

$$
AUC=0.468670<0.5,
$$

$$
Precision=0.238095,
$$

$$
StepRatio_{q90}=23.828453.
$$

这说明 source-logged CP5 不能直接变成 online controller。即使某些 horizon-consistency 信息是可重算的，当前 legal probe 的实现没有保留其分类能力，而且系统成本完全不可接受。

### 0.3 当前不能怎样解释

不能解释为：

```text
functional update 彻底失败；
v8-FT7 functional core 是假的；
strict FC-PureKAN base 不可训练；
数据集需要分别调参；
再调一个 target / threshold 就能过。
```

更准确的解释是：

$$
\boxed{
\text{current functional primitive 的 causal value 还不能在 commit 前被低成本、合法、稳定观测。}
}
$$

也就是说，我们现在不是缺 “更复杂的 dataset-specific controller”，而是缺 **primitive-level observability**。

### 0.4 当前最重要的新边界

v9.2.33 之后，问题已经从：

```text
能不能设计一个 event-value predictor？
```

推进为：

```text
当前 primitive 是否提供了足够可观测的 causal sufficient statistic？
```

因为 legal probe 已经真实测过：

```text
not predictive
too expensive
```

而 OP1-OP6 仍未实现。下一步如果继续在 OP0 / N2a / N2c / N3c 上调 probe，很可能只是继续重复同一个失败。v9.2.34 必须真正实现 observable primitive family。

---

## 1. v9.2.34 总体目标

v9.2.34 的总体目标是：

$$
\boxed{
\text{实现 OP1-OP6 observable primitives，并证明至少一个 primitive 让 functional causal value 在 commit 前可观测。}
}
$$

本轮不是 full-run 目标，而是一个更基础的 primitive/interface closure 目标。它要完成四件事。

### 1.1 Observable primitive implementation success

至少一个 OP primitive 必须满足：

```text
edge-owned params only
manual forward = 1
manual backward = 1
manual update = 1
uses_loss_backward = 0
external_residual_used = 0
ordinary_mlp_path_used = 0
GradRelErrMax <= 1e-4
GradCosMin >= 0.999
P4 pass
P5 near-pass
functional actuatability pass
observability pass
```

### 1.2 Observability success

Observable primitive 的关键不是 movement 更大，而是 movement 的 value 可预测。成功标准为：

$$
AUC(Y_{\text{beat}})\geq0.70,
$$

或：

$$
Corr(S_{\text{obs}},V_{\text{grounded}})\geq0.35.
$$

其中：

$$
Y_{\text{beat}}
=
\mathbb{1}
[
Gain_{\text{Real}}
>
\max(Gain_{\text{AdamWParallel}}, Gain_{\text{bestLR}})
].
$$

并且 accepted events 必须满足：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 1.3 System success

Observable primitive 不能靠巨额 probe 成本过关。Official candidate 必须满足：

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

如果 observability 过但 system 不过，则 route 只能是：

```text
ObservableButTooExpensive
```

不能进入 paired replay。

### 1.4 Local functional causality success

只有 observable primitive 通过后，才能进入 official paired replay。paired replay 成功标准：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

并且每个 evaluation slice task-safe：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

注意这里的 slice 可以是 dataset slice，但 official controller 不能使用 dataset name。数据集只能诊断，不允许调参。

---

## 2. 核心假设

### H1：v9.2.33 legal probe 失败不是因为 functional value 不存在，而是 current primitive 不可观测

v9.2.31 已经证明 grounded event value 可靠。v9.2.32 的 source-logged CP5 也显示过强分类信号。但 v9.2.33 legal probe 失败，说明 current primitive 的 value 不容易被 legal online probe 读出。

H1 成立标准：

如果 OP primitive 能达到：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

或：

$$
Corr(S_{\text{obs}},V_{\text{grounded}})\geq0.35,
$$

则说明原问题主要是 primitive observability，而不是 functional concept 失败。

### H2：observable primitive 的目标是暴露 causal sufficient statistic，而不是增大 output displacement

过去多轮已经证明：

```text
actuatability 大 != causal gain
source-logged signal != legal signal
Real effect != control-resistant effect
```

因此 OP primitive 设计不能以 $r_z$ 最大为目标，而要让下面的量可计算、稳定、低噪：

$$
S_{\text{obs}}
\approx
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

H2 成立标准：

若某 primitive 的 movement ratio 不最大，但它的 $S_{\text{obs}}$ 对 grounded value 的 AUC / corr 最高，则优先选该 primitive，而不是选 movement 最大者。

### H3：tail-local / orthogonal-tail primitive 比 global rational primitive 更可能可观测

当前 OP0 / N2a rational reference corr 只有约 `0.134571`。这提示 global rational channel 的影响太分散，pre-commit value 难估计。  
H3 认为 local primitive 更可能形成可观测 sufficient statistic：

```text
piecewise local tail channel
shared-RBF local channel
orthogonal tail channel
control-gap channel
```

H3 成立标准：

至少一个 local / orthogonal primitive 的 observability 显著高于 OP0：

$$
AUC_{\text{OP-local}}
-
AUC_{\text{OP0}}
\geq0.10,
$$

或：

$$
Corr_{\text{OP-local}}
-
Corr_{\text{OP0}}
\geq0.10.
$$

### H4：official controller 必须 signal-based，不能 dataset-based

允许诊断：

```text
MNIST = easy-mode / abstention safety slice
Fashion-MNIST = delayed / control-dominated tail slice
KMNIST = hard-mode / effect magnitude slice
```

但 official route 禁止：

```text
if dataset == Fashion: ...
if dataset == KMNIST: ...
if dataset == MNIST: ...
```

H4 成立标准：

Leave-dataset-out validation 必须通过：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out splits 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50.
$$

### H5：如果 OP1-OP6 都失败，则应进入 deeper primitive/interface redesign，而不是继续调 controller

如果 OP1-OP6 都无法同时满足 P4/P5/observability，则说明 current edge-function interface family 不能承载可观测 functional update。下一步应回到 primitive/interface design，甚至重新借鉴 v8-FT7 的 role-wise mechanism，而不是继续 dataset patch。

H5 触发标准：

```text
all OP1-OP6 fail observability
or all OP1-OP6 fail P4/P5
or paired replay remains control-equivalent after observability pass
```

---

## 3. Candidate 设计

## 3.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  strict FC-PureKAN AdamW-only base。

OP0-current-reference:
  current N2a/N2c/N3c family reference。

V8-FT7:
  mechanism source only，不作为 official strict candidate。
```

## 3.2 Observable primitives

### OP1：ObservableTailLinearChannel

目标：建立一个最简单、解析可观测的 tail-local edge-owned channel。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
a_{ij}\cdot q(h)\cdot h.
$$

其中 $q(h)$ 是 signal-defined gate，不使用 dataset name。第一版取 smooth bounded gate：

$$
q(h)=\sigma(\gamma(|h|-\tau)).
$$

可观测统计：

$$
S_{\text{obs}}
=
\langle
\Delta z_{\text{tail}},
-
\nabla_z CE_{\text{tail}}
\rangle
-
\max(S_{\text{AdamWParallel}},S_{\text{bestLR}}).
$$

优点是 analytic logit movement 可直接估计；缺点是表达力可能弱。

### OP2：ObservablePiecewiseTailChannel

目标：用局部 piecewise basis 捕捉 hard-tail local correction，同时保持 derivative bounded。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
\sum_{k=1}^{K}
a_{ij,k}
\cdot
\operatorname{clip}(h-\tau_k,0,\Delta).
$$

默认：

```text
K = 2,4
shared knots
no B-spline recursion
bounded slope
```

可观测统计：

$$
S_{\text{obs}}
=
\sum_k
\langle
\Delta z_{\text{tail},k},
-\nabla_z CE_{\text{tail}}
\rangle
-
S_{\text{control}}.
$$

### OP3：ObservableSharedRBFLocalChannel

目标：用 shared-center RBF 局部通道，让 event effect 在 center activation 上可观测。

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

默认：

```text
K = 4
shared centers
normalized basis energy
no per-edge center materialization
```

可观测统计：

$$
S_{\text{obs}}
=
\sum_k
E_{\text{tail},k}
\cdot
\Delta a_k
-
S_{\text{control}},
$$

其中 $E_{\text{tail},k}$ 是 tail samples 上第 $k$ 个 basis 的 normalized activation energy。

### OP4：ObservableOrthogonalTailChannel

目标：显式产生非 AdamW 平行的 tail correction，避免 RealFunctional 只是 AdamWParallel / LR disguise。

定义 candidate update：

$$
\Delta\theta_{\perp}
=
\Delta\theta_{\text{func}}
-
\operatorname{proj}_{\Delta\theta_{\text{AdamW}}}
\Delta\theta_{\text{func}}.
$$

只保留 tail-relevant component：

$$
\Delta\theta_{\text{OP4}}
=
P_{\text{tail}}\Delta\theta_{\perp}.
$$

可观测统计：

$$
r_{\perp,\text{tail}}
=
\frac{
\|\Delta z_{\perp,\text{tail}}\|
}{
\|\Delta z_{\text{AdamW},\text{tail}}\|+\epsilon
}.
$$

OP4 pass 不能只看 $r_{\perp,\text{tail}}$，还必须看 $V_{\text{grounded}}$。

### OP5：ObservableControlGapChannel

目标：把 Real-vs-control gap 直接写入 primitive 的可观测 lower bound。

构造：

$$
S_{\text{gap-lb}}
=
LB(Gain_{\text{Real}})
-
UB(Gain_{\text{control}}).
$$

只在：

$$
S_{\text{gap-lb}}>0
$$

时允许 functional update。

这个 primitive 可能保守，但如果 precision 高且 coverage 不低于 $0.03$，可以成为 official candidate。

### OP6：LightHybridObservable

目标：保留 N2a rational 的 global smooth signal，同时加一个 local observable tail channel。

Edge function：

$$
\phi_{ij}(h)
=
\phi^{task}_{ij}(h)
+
a^{r}_{ij}\frac{h^2}{1+\beta h^2}
+
\sum_k a^{l}_{ij,k}B_k(h).
$$

其中 $B_k$ 可以是 piecewise 或 shared RBF。  
OP6 只有在 rational channel 和 local channel 都可审计、P4/P5 不破坏时才能进入 official route。

## 3.3 Probe / controller candidates

OP primitive 下不再做 source-logged CP5。只允许 legal train-stream statistics：

```text
C0-NoFunctional
C1-OPValueScore
C2-OPValueScore+UncertaintyLCB
C3-OPValueScore+ControlContrastive
C4-OPValueScore+HorizonConsistency
C5-OPValueScore+OrthogonalTail
```

Controller 不能使用：

```text
dataset_name
seed_id
test metric
validation metric
posthoc replay outcome at commit
class-specific hand rule
```

## 3.4 Controls

每个 official replay 必须包括：

```text
AdamWOnly
AdamWParallelTrustRatio-0.003
AdamWParallelTrustRatio-0.01
AdamWParallelTrustRatio-0.03
BestLRScale
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledRoleMask
InvertedRoleMask
FrozenFuncChannel
ShuffledFuncChannel
DatasetRouteShuffled
EventRouteShuffled
SeverityBucketShuffled
SignalScoreShuffled
PreMovementShuffled
ControlGapShuffled
ProbeScoreShuffled
PrimitiveScoreShuffled
TailMaskShuffled
```

新增 controls：

```text
PrimitiveScoreShuffled:
  keep event and primitive fixed, shuffle primitive observability score。

TailMaskShuffled:
  keep update norm, shuffle tail mask assignment。
```

---

## 4. 实验阶段

## P0：v9.2.33 boundary reproduction

### 目标

复现 v9.2.33 boundary，确认当前不是 measurement artifact。

### 必须记录

```text
route
source_route_v9232
cp5_legality_classification
cp5_best_feature_auc
best_legal_probe
legal_probe_corr
legal_probe_auc
legal_probe_precision
legal_probe_coverage
legal_probe_bad_event_rate
legal_probe_step_q90
legal_probe_memory_ratio
legal_probe_predictive_pass
legal_probe_system_pass
best_observable_primitive
best_observable_primitive_corr
op_not_implemented_count
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R14-ReturnToInterfacePrimitiveDesign
legal_probe_predictive_pass = 0
legal_probe_system_pass = 0
OP1-OP6 not implemented count = 6
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_legal_probe_failure.svg
p0_op_implementation_gap.svg
```

---

## P1：Legal probe failure autopsy

### 目标

解释 source CP5 到 legal probe 的断裂，防止重复失败。

### 必须记录

```text
probe
source_or_legal
feature_source
available_before_commit
posthoc_component_fraction
train_stream_recomputable
probe_split
horizon
corr
auc
precision
coverage
bad_event_rate
step_q90
memory_ratio
failure_reason
```

### Failure taxonomy

```text
L1-source_posthoc_leak:
  source CP5 AUC 来自 posthoc-only rows。

L2-legal_horizon_mismatch:
  legal horizon score 与 source horizon score 分布不同。

L3-probe_split_noise:
  update/probe split 噪声过大。

L4-control_gap_label_noise:
  Real-vs-control label 高方差。

L5-system_infeasible:
  predictive signal 有，但 step/memory 不可接受。

L6-current_primitive_unobservable:
  legal probe 和 static feature 都低相关。
```

### 判断标准

P1 pass：

```text
每个 legal probe row 都归入一种 primary failure reason；
source CP5 的合法可复算部分与 posthoc 部分分离；
明确是否值得继续优化 probe。
```

如果 P1 显示 legal probes 的主要失败是 `L6-current_primitive_unobservable`，则 P2 直接进入 OP implementation，不再做 probe tuning。

### 可视化

```text
p1_source_vs_legal_score_distribution.svg
p1_probe_failure_taxonomy.svg
p1_horizon_score_mismatch.svg
p1_probe_cost_vs_predictivity.svg
```

---

## P2：Observable primitive implementation

### 目标

真正实现 OP1-OP6，不允许继续记录为 not_implemented。

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
basis_count
shared_basis
bounded_derivative
analytic_observability_stat
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
OP1-OP6 至少实现 4 个；
每个实现候选必须能 forward / backward / update smoke；
未实现项必须有明确技术原因，不得默认为 route candidate。
```

### 可视化

```text
p2_op_implementation_matrix.svg
p2_observable_stat_by_primitive.svg
```

---

## P3：Contract / gradcheck / interaction audit

### 目标

证明 OP primitives 是 strict PureKAN，并且梯度正确。

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
functional_channel_entropy
dominant_basis_fraction
```

### 判断标准

Contract pass：

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward/backward/update = 1
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

如果某 OP 是 deliberately local 而非 pairwise-global，则要额外记录 local-bump R2，不能用低 pairwise R2 直接判死；但 official FullEdge route 仍必须说明它如何保留 task expressivity。

### 可视化

```text
p3_contract_grad_matrix.svg
p3_pairwise_local_fit.svg
p3_basis_conditioning.svg
```

---

## P4：P4/P5 base qualification

### 目标

保证 OP primitive 不破坏 base system 与 AdamW-only trainability。

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

Full-pass diagnostic：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

### 可视化

```text
p4_system_pareto.svg
p4_task_nearpass_matrix.svg
p4_base_gap_by_primitive.svg
```

---

## P5：Primitive observability audit

### 目标

检查 OP primitive 是否让 event value 在 commit 前可观测。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240
events = signal strata S1-S8
controllers = C1-C5
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
primitive
controller
event_id
signal_stratum
horizon
obs_score
obs_score_tail
obs_score_orthogonal
obs_control_gap
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
Corr(S_{\text{obs}},V_{\text{grounded}})\geq0.35.
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
p5_obs_score_vs_grounded_value.svg
p5_auc_precision_coverage_by_primitive.svg
p5_obs_system_pareto.svg
p5_accepted_strata_distribution.svg
```

---

## P6：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 OP controller 隐式 dataset tuning。

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

## P7：Official observable-primitive paired replay

### 目标

只允许 P2-P6 survivor 进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
primitive = best OP survivor
controller = best legal signal-based controller
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, PrimitiveScoreShuffled, TailMaskShuffled, SignalScoreShuffled
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, PrimitiveScoreShuffled
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

## 5. Required artifacts

```text
run_manifest.json
contract_audit_v9234.csv
p0_v9233_boundary_reproduction.csv
p1_legal_probe_failure_autopsy.csv
p2_observable_primitive_implementation.csv
p3_contract_grad_interaction_audit.csv
p4_op_p4_p5_base_qualification.csv
p5_primitive_observability_audit.csv
p6_leave_dataset_and_stratum_out_validation.csv
p7_official_observable_primitive_paired_replay.csv
p8_short_run_functional_validation.csv
p9_full_10seed_functional_validation.csv
p10_adamw_only_fullpass_repair.csv
p11_robustness_external_ready.csv
observable_primitive_trace_v9234.csv
observability_score_trace_v9234.csv
leave_dataset_out_trace_v9234.csv
paired_replay_branch_trace_v9234.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9233_boundary_unstable
F3_dataset_tuning_detected
F4_source_cp5_posthoc_only
F5_legal_probe_failure_unattributed
F6_op_not_implemented
F7_op_contract_fail
F8_op_grad_fail
F9_op_p4_fail
F10_op_p5_fail
F11_op_observability_fail
F12_op_system_too_expensive
F13_leave_dataset_out_fail
F14_leave_stratum_out_fail
F15_paired_replay_control_equivalent
F16_shuffle_control_pass
F17_functional_lr_equivalent
F18_short_run_task_drop
F19_full_run_no_macro_hard_stratum_gain
F20_adamw_fullpass_fail
F21_strong_baseline_explains_gain
F22_robustness_fail
F23_external_not_ready
F24_fake_or_proxy_violation
F25_artifact_missing
```

---

## 6. Route decision

### Route cases

```text
R1-LegalProbeFailureAttributed:
  source-to-legal probe gap is explained.

R2-ObservablePrimitiveImplemented:
  at least four OP primitives implemented and audited.

R3-ObservablePrimitiveContractPass:
  at least one OP primitive passes contract / grad / interaction.

R4-ObservablePrimitiveBaseQualified:
  at least one OP primitive passes P4 and P5 near-pass.

R5-ObservablePrimitiveObservabilityPass:
  OP primitive exposes commit-time value signal.

R6-LeaveDatasetOutOPPass:
  OP controller generalizes without dataset-specific tuning.

R7-ObservablePrimitivePairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R8-ObservableButTooExpensive:
  observability passes but system envelope fails.

R9-OPAllFailPrimitiveReset:
  OP1-OP6 fail implementation / contract / observability.

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
v9233_boundary_pass
dataset_tuning_detected
legal_probe_failure_mode
op_implemented_count
best_op_candidate
op_contract_pass
op_grad_pass
op_p4_pass
op_p5_nearpass
op_observability_pass
op_system_pass
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
success_v9234_strict_purekan_functional
success_v9234_full_functional
success_v9234_external_ready
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.33 boundary。

Step 2:
  P1 做 legal probe failure autopsy。
  确认 source CP5 到 legal probe 断裂的原因。

Step 3:
  P2 实现 OP1-OP6。
  不允许继续把 observable primitives 写成 not_implemented。

Step 4:
  P3 做 contract / gradcheck / interaction audit。
  不是 PureKAN 的 candidate 直接剔除。

Step 5:
  P4 做 P4/P5 base qualification。
  不能用 functional update 救 base。

Step 6:
  P5 做 primitive observability audit。
  重点看 commit 前的可观测 value，而不是 movement 大小。

Step 7:
  P6 做 leave-dataset-out / leave-stratum-out。
  防止隐式 dataset tuning。

Step 8:
  P7 official paired replay。
  只有 OP survivor 才能进入。

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

## 8. 停止条件

### Minimum diagnostic success

```text
v9.2.33 boundary reproduced
legal probe failure attributed
OP1-OP6 at least 4 implemented
at least one OP passes contract / grad
no fake/proxy/offload/loss/teacher violation
```

### Observable primitive success

```text
Minimum diagnostic success
+
at least one OP passes P4/P5 near-pass
+
observability pass
+
leave-dataset-out pass
```

### Local functional success

```text
Observable primitive success
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
1. v9.2.33 boundary cannot be reproduced；
2. legal probe failure cannot be attributed；
3. OP1-OP6 cannot be implemented as strict PureKAN；
4. all OP candidates fail gradcheck；
5. all OP candidates fail P4/P5；
6. all OP candidates fail observability；
7. OP observability passes but system envelope fails；
8. leave-dataset-out fails；
9. paired replay remains control-equivalent；
10. shuffle controls pass, indicating overfit；
11. full run gives no macro / hard-stratum / geometry gain；
12. functional breaks system gate；
13. gains are explained by QuadraticFeatureMLP；
14. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：OP observability passes

可以声明：

```text
current blocker was primitive observability, and a strict PureKAN observable primitive now exposes commit-time functional value.
```

但不能声明 full functional success，除非 P7-P9 也通过。

### Case B：OP passes observability but fails system

必须声明：

```text
functional value is observable, but current primitive is not kernel-native enough.
```

下一步转系统 kernelization，而不是 paired replay。

### Case C：OP passes P4/P5 but fails observability

必须声明：

```text
the primitive is trainable and efficient, but still does not expose functional causal value.
```

下一步回到 primitive/interface design，而不是 target patch。

### Case D：leave-dataset-out fails

必须声明：

```text
controller is implicitly dataset-specific; official success is not allowed.
```

不能用 dataset tuning 写成功。

### Case E：all OPs fail

必须声明：

```text
current observable primitive family cannot carry dataset-agnostic functional update.
```

下一步应回到 deeper edge-function interface design，尤其重新抽取 v8-FT7 的 role-wise mechanism，而不是继续 Fashion/KMNIST patch。

---

## 10. 最终建议

v9.2.34 的一句话策略是：

$$
\boxed{
\text{停止继续修 legal probe；真正实现 observable primitives，并以 observability 而不是 displacement 作为晋级门。}
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
1. 什么样的 edge-owned functional channel 能让 causal value 在 commit 前可观测？
2. 这种 observability 是否能在不使用 dataset name 的情况下泛化？
3. 这种 primitive 是否仍然 P4/P5 合格？
4. RealFunctional 是否能在 official paired replay 中超过 AdamWParallel / best LR？
5. 如果 OP1-OP6 都不行，是否要重新从 v8-FT7 role-wise mechanism 设计 strict PureKAN primitive？
```
