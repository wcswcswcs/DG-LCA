# DG-KAN v9.2.22 Base-Qualified Strict PureKAN Functional Interface：从 I1c P5 断点到可迁移 Functional Core 的完整实验计划

> 本计划基于 v9.2.21 `Strict PureKAN Functional Interface Transfer` 的真实 terminal route 制定。  
> v9.2.21 已经把问题推进到一个非常明确的新边界：**v8-FT7 functional core 已经被保留并抽取出机制；strict PureKAN dual-role interface 的 contract / gradcheck / P4 也已经可做；但当前 best interface `I1c-T2Task-RationalFunc-DerivativeControlled` 没有通过 P5 base near-pass。**
>
> 因此 v9.2.22 不再继续手写 output target，不再继续调 SNR threshold，不再继续把 paired replay 越过 P5 gate，也不提前进入 PureKANConv / PureKANFormer。  
> 本轮核心问题是：
>
> $$
> \boxed{
> \text{如何让 strict PureKAN functional interface 在不破坏 AdamW base 的前提下承载 v8-FT7 functional core？}
> }
> $$
>
> 本轮继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal / margin / calibration loss、no sampler / class weight、no CPU offload、no fake / proxy rows、KAN path 不使用 PyTorch loss.backward graph。**

---

## 0. 当前结果的独立判断

### 0.1 v9.2.21 达成了什么

v9.2.21 的 terminal route 是：

```text
route = R7-InterfaceBreaksBase
base_candidate = LQ-t2-h256
success_v9221_strict_purekan_functional = false
success_v9221_full_functional = false
success_v9221_external_ready = false
```

这不是 functional core 的失败。恰恰相反，v9.2.21 证明了两件非常重要的事。

第一，v8-FT7 functional core 的机制信号已经从 v9.2.20 full replay 中被抽取出来：

```text
functional family corr effective derivative vs -curvature = 0.992452
functional family corr branch ratio vs -curvature = 0.995620
FT7 mean cosine with AdamW = -0.049826
FT7 beats AdamWParallel = 1.000000
FT7 beats best LR = 1.000000
```

这说明 FT7 的核心不是 scalar LR，也不是 AdamWParallel disguise；它有非 AdamW、role-wise、derivative/branch-ratio 驱动的 curvature 机制。

第二，strict PureKAN interface 的系统与正确性不是不可做。P2/P3 已经证明多个 edge-owned interface 通过了 contract、gradcheck、interaction 和 P4，其中 `I1b/I1c/I3a` 都进入 P4 gate。当前 best 是：

```text
I1c-T2Task-RationalFunc-DerivativeControlled
actuator = A4e-BoundedRational-FusedCoeffGrad
interface_contract_pass = 1
interface_p4_pass = 1
```

### 0.2 v9.2.21 没有达成什么

v9.2.21 没有通过 P5 base qualification，因此 paired replay、short-run、full-run、robustness、external-ready 都没有打开。  
`I1c` 的 P5 base qualification 是：

```text
p5_near_pass_count = 7
p5_row_count = 9
p5_macro_delta = -0.004777789115905762
interface_p5_nearpass = 0
```

失败 rows 是：

```text
MNIST seed2:
  KAN acc = 0.942500
  MLP-match acc = 0.953000
  delta = -0.010500
  near-pass = 0

KMNIST seed0:
  KAN acc = 0.822000
  MLP-match acc = 0.836500
  delta = -0.014500
  near-pass = 0
```

这两个 failure row 很关键。它们说明问题不是整体 collapse，也不是 P4 系统失败，而是 **functional channel 加入后破坏了 base trainability 的局部稳定性**。如果现在直接打开 P4 paired replay，用 functional 去补 P5 failure，就会回到以前的错误：用 functional 掩盖一个还未 base-qualified 的 primitive。

### 0.3 当前最重要的判断

当前 blocker 不是：

```text
functional core 不存在；
FT7 是 LR disguise；
strict interface contract 做不到；
manual backward / gradcheck 不过；
P4 system gate 不过；
actuator 不可控；
output target 不存在。
```

当前 blocker 是：

$$
\boxed{
\text{strict functional channel 介入 base training 后，破坏了 P5 near-pass 稳定性。}
}
$$

更深层地说，v9.2.21 暴露的是 **task channel 与 functional channel 没有正确解耦**。  
FT7 的 functional core 是一个 event/update interface；而 I1c 把 rational functional channel 直接放进 base model 中参与 AdamW base qualification。这样做可能让 functional channel 提前参与 task fitting，导致：

```text
1. task channel 和 functional channel 竞争；
2. rational derivative scale 影响 early optimization；
3. functional channel 在没有 functional event 时也改变 base trajectory；
4. branch ratio / derivative scale 没有被限制在 FT7 的安全机制区间；
5. 少数 seed / hard-mode rows 被放大成 near-pass failure。
```

因此 v9.2.22 的重点不是“再换一个 target”，而是 **构造 base-neutral 的 strict PureKAN functional interface**。

---

## 1. v9.2.22 总体目标

v9.2.22 的总体目标是：

$$
\boxed{
\text{获得一个 P4-qualified、P5-nearpass、且具备 non-LR functional actuatability 的 strict PureKAN interface。}
}
$$

它必须同时满足三层条件。

### 1.1 Base qualification

当 functional update 关闭时，strict PureKAN interface 不能破坏 AdamW-only base：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

并且最好恢复到或超过 LQ/A7c 的稳定性：

$$
near\_pass\_count\geq8/9.
$$

### 1.2 Functional interface qualification

Functional channel 必须不是普通 LR 或 AdamWParallel disguise。它必须在 paired replay 中满足：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

并且至少一个机制指标成立：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

### 1.3 System qualification

所有 official candidate 必须继续保持 P4：

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

---

## 2. 核心假设

### H1：I1c 的 P5 failure 来自 functional channel 过早参与 base fitting

I1c 的 rational functional channel 在 functional_update off 时仍可能通过 AdamW 被训练并影响 base trajectory。  
这会破坏 task channel 的原有 LQ 稳定性。

H1 成立标准：

如果将 functional channel 设置为 base-neutral 后：

```text
near-pass count 从 7/9 提升到 >= 8/9；
MNIST seed2 或 KMNIST seed0 至少一个被修复；
P4 仍 pass；
functional actuatability 不消失；
```

则 H1 成立。

Base-neutral 可以包括：

```text
zero-init functional channel；
AdamW base phase 冻结 functional channel；
functional-only update mask；
functional channel 不参与 base optimizer state；
delayed functional channel activation；
task-channel warm-start followed by edge-owned functional attach。
```

### H2：FT7 机制需要 branch-ratio / derivative-scale 安全区间，而不是无约束 rational channel

v9.2.21 的 P1 显示 branch ratio 与 effective derivative scale 和 curvature gain 高相关。  
因此 strict interface 不能只追 target-fit 或 P4；必须把 branch ratio 与 derivative scale 纳入资格门。

定义：

$$
r_{\text{branch}}
=
\frac{\|\phi^{func}(h)\|}{\|\phi^{task}(h)\|+\epsilon}.
$$

定义：

$$
s_{\text{eff}}
=
|\alpha|\cdot B'_a(h)_{p95}.
$$

H2 成立标准：

存在安全区间：

$$
r_{\min}\leq r_{\text{branch}}\leq r_{\max},
$$

$$
s_{\min}\leq s_{\text{eff}}\leq s_{\max},
$$

使得：

```text
bad_event_rate <= 0.05；
paired replay beats AdamWParallel / best LR；
P5 base near-pass 不下降。
```

### H3：task channel 与 functional channel 必须在 optimizer role 上解耦

Functional channel 可以存在于 edge function 内，但它不应该被 base AdamW 当作普通 task parameter 无差别更新。  
严格形式为：

$$
\theta =
(\theta_{\text{task}},\theta_{\text{func}}).
$$

Base phase：

$$
\theta_{\text{task},t+1}
=
\theta_{\text{task},t}
+
\Delta\theta_{\text{AdamW}},
$$

$$
\theta_{\text{func},t+1}
=
\theta_{\text{func},t}.
$$

Functional event phase：

$$
\theta_{\text{func},t+1}
=
\theta_{\text{func},t}
+
\Delta\theta_{\text{FT7-like}},
$$

while $\theta_{\text{task}}$ still follows AdamW.

这不是 external residual，也不是 loss modification；它是 edge-owned parameter role mask。

H3 成立标准：

role-decoupled candidate 满足：

```text
P4 pass；
P5 near-pass；
functional channel nonzero event displacement；
paired replay beats AdamWParallel / best LR。
```

### H4：KMNIST seed0 是 functional-interface stress test，MNIST seed2 是 base-stability stress test

I1c 的两个 miss row 不同：

```text
MNIST seed2:
  easy dataset 上掉到 -0.0105，说明 base optimization / initialization 失稳；

KMNIST seed0:
  hard-mode 上掉到 -0.0145，说明 functional channel / rational tail 可能放大了 hard-mode instability。
```

H4 成立标准：

P1 autopsy 能把两个 miss row 分别归因到：

```text
branch ratio；
derivative p95；
functional channel usage entropy；
lift condition number；
CEp99 / margin tail；
wrong confidence；
cos with task gradient；
```

并且 P2 repair 后至少一个 failure mode 被修复。

### H5：如果 base-neutral interface 仍不能 P5 near-pass，则当前 dual-role rational interface family 不适合作为 strict transfer path

如果 I1/I2/I3/I4/I5 系列在 base-neutral / role-decoupled / derivative-controlled 后仍无法达到 P5 near-pass，则说明当前 interface family 本身不适配。  
下一步应回到 basis / primitive factory，而不是继续调这个 rational channel。

---

## 3. Candidate 设计

### 3.1 Baselines

```text
B0-MLP-match:
  official same-parameter MLP baseline。

LQ0-LQ-t2-h256:
  current strict FC-PureKAN base。

A7c-BasisEntropy-ValueOnly:
  P4-qualified actuator reference。

I1c-current:
  T2 task + rational functional derivative-controlled current best, P4 pass, P5 7/9.
```

### 3.2 Base-neutral strict interface candidates

#### N1：Zero-impact functional channel

Functional channel zero-init，并在 AdamW base phase 冻结：

$$
\phi_{jc}(h)
=
\phi^{task}_{jc}(h)
+
\phi^{func}_{jc}(h),
$$

其中：

$$
\phi^{func}_{jc}(h)=0
$$

at initialization and remains frozen during base AdamW qualification.

Candidates：

```text
N1a-ZeroInit-RationalFunc-FrozenBase
N1b-ZeroInit-PiecewiseFunc-FrozenBase
N1c-ZeroInit-SharedRBFFunc-FrozenBase
N1d-ZeroInit-CenteredT2Func-FrozenBase
```

#### N2：Tiny-impact functional channel

Functional channel 使用 tiny init，但 branch ratio 被限制：

$$
r_{\text{branch}}\leq r_{\max}.
$$

Candidates：

```text
N2a-TinyInit-RationalFunc-BranchRatioCap
N2b-TinyInit-PiecewiseFunc-BranchRatioCap
N2c-TinyInit-SharedRBFFunc-BranchRatioCap
```

#### N3：Derivative-scale controlled channel

Functional channel 不一定 zero，但 derivative p95 被校准到 FT7-like safe band：

$$
s_{\text{eff}}
=
|\alpha|\cdot B'_a(h)_{p95}.
$$

Candidates：

```text
N3a-RationalFunc-DerivativeBand
N3b-PiecewiseFunc-DerivativeBand
N3c-SharedRBFFunc-DerivativeBand
```

#### N4：Role-decoupled optimizer channel

Functional channel 可以存在，但 base AdamW 不更新它；只在 functional event 中更新：

```text
AdamW mask:
  task channel = trainable
  functional channel = frozen

Functional mask:
  task channel = frozen or projected
  functional channel = trainable
```

Candidates：

```text
N4a-RationalFunc-AdamWFrozen-FunctionalOnly
N4b-PiecewiseFunc-AdamWFrozen-FunctionalOnly
N4c-SharedRBFFunc-AdamWFrozen-FunctionalOnly
```

#### N5：Task-warmstart functional attach

先训练 LQ task channel 到 base checkpoint，然后 attach zero-impact functional channel 并从该 checkpoint 进入 functional paired replay。  
该方案必须写清楚：attach 的 parameters 仍是 edge-owned，只是训练 schedule 不同，不是 external residual。

Candidates：

```text
N5a-LQWarmstart-RationalFuncAttach
N5b-LQWarmstart-PiecewiseFuncAttach
N5c-LQWarmstart-SharedRBFFuncAttach
```

### 3.3 Controls

所有 P4/P5/Paired Replay 必须包括：

```text
AdamWOnly
AdamWParallelSameNorm
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
```

---

## 4. 实验阶段

## P0：v9.2.21 boundary reproduction

### 目标

复现当前 terminal boundary，确认不是测量噪声。

### 必须记录

```text
route
best_interface_candidate
interface_contract_pass
interface_p4_pass
interface_p5_nearpass
p5_near_pass_count
p5_row_count
p5_macro_delta
failed_rows
no_fake_proxy
```

### 判断标准

P0 pass：

```text
route = R7-InterfaceBreaksBase
best candidate = I1c
P4 pass = 1
P5 near-pass = 0
near-pass count = 7/9 ± tolerance
fake/proxy = 0
```

### 可视化

```text
p0_v9221_boundary_dashboard.svg
p0_i1c_failed_rows.svg
p0_p4_p5_gate_summary.svg
```

---

## P1：I1c P5 failure autopsy

### 目标

解释 I1c 为什么 P4 pass 但 P5 near-pass fail。  
P1 不新增 candidate，只分析 current I1c 与 LQ/A7c 的差异。

### 必须记录

```text
candidate
dataset
seed
KAN_acc
MLP_match_acc
delta
near_pass
CEp50
CEp90
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
logit_norm_p95
functional_channel_output_norm
task_channel_output_norm
branch_ratio
effective_derivative_p95
functional_channel_usage_entropy
dominant_functional_basis_fraction
lift_condition_number
effective_rank
cos_func_channel_task_gradient
cos_func_channel_adamw_update
update_norm_task_channel
update_norm_func_channel
```

### 判断标准

P1 必须把 failure 归入至少一种：

```text
B1-functional_channel_competes_with_task:
  update_norm_func_channel / update_norm_task_channel 高，且 miss rows branch_ratio 高。

B2-derivative_scale_unstable:
  effective_derivative_p95 在 miss rows 明显高于 pass rows。

B3-functional_channel_underused:
  usage entropy 低，dominant basis fraction 高，functional channel 没有形成稳定接口。

B4-lift_conditioning_regression:
  lift condition number 或 effective rank 明显劣化。

B5-hard_tail_amplification:
  CEp99 / wrong_confidence / margin tail 在 KMNIST seed0 明显恶化。

B6-base_init_seed_sensitivity:
  MNIST seed2 与初始化 / early trajectory 有关，而非 hard-mode。
```

### 可视化

```text
p1_failure_autopsy_heatmap.svg
p1_branch_ratio_vs_delta.svg
p1_derivative_p95_vs_delta.svg
p1_functional_channel_usage_entropy.svg
p1_failed_rows_ce_margin_panel.svg
```

---

## P2：Base-neutral interface factory

### 目标

构造不会破坏 AdamW base 的 strict PureKAN functional interface。  
P2 的成功不是 functional success，而是让 interface 重新获得 P5 qualification。

### 设置

```text
candidates = N1a-N5c
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off for P5 base qualification
baseline = MLP-match
```

### 必须记录

```text
candidate
interface_family
basis_formula
functional_channel_init
functional_channel_adamw_trainable
functional_channel_event_trainable
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
GradRelErrMax
GradCosMin
pairwise_R2
forward_q90
backward_q90
step_q90
memory_ratio
dataset
seed
KAN_acc
MLP_match_acc
delta
near_pass
CEp99
margin_p10
ECE
NLL
branch_ratio
effective_derivative_p95
functional_channel_usage_entropy
```

### 判断标准

Contract pass：

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward/backward/update = 1
```

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

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
near\_pass\_count\geq8/9,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Promotion preference：

```text
repair MNIST seed2 or KMNIST seed0
and preserve functional channel actuatability
```

### 可视化

```text
p2_base_neutral_interface_pareto.svg
p2_p5_nearpass_by_candidate.svg
p2_failed_rows_repair_matrix.svg
p2_task_system_functional_interface.svg
```

---

## P3：Functional actuatability audit for P5-qualified candidates

### 目标

P5-qualified interface 还不能直接进入 full functional。  
P3 需要证明它的 functional channel 仍有 non-LR actuatability。

### 必须记录

```text
candidate
dataset
seed
event_type
functional_channel
target_fit_R2
output_displacement_ratio
non_adamw_output_displacement_ratio
r_perp
cos_with_adamw
branch_ratio_after_event
effective_derivative_after_event
bad_event_rate
holdout_nonharm
```

### 定义

Non-AdamW output displacement：

$$
r_{z,\perp}
=
\frac{
\|\Delta z_{\text{func}}-\operatorname{proj}_{\Delta z_{\text{AdamW}}}\Delta z_{\text{func}}\|
}{
\|\Delta z_{\text{AdamW}}\|+\epsilon
}.
$$

Actuatability pass：

$$
r_{z,\perp}\geq0.05,
$$

and:

$$
bad\_event\_rate\leq0.05.
$$

### 可视化

```text
p3_non_adamw_actuatability.svg
p3_rperp_by_candidate.svg
p3_branch_ratio_after_event.svg
p3_bad_event_rate.svg
```

---

## P4：Paired replay control gate

### 目标

只有 P2/P3 survivor 才进入 P4。  
验证 strict PureKAN interface 是否真正超过 AdamWParallel / LR / Random / NoOp controls。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80,240,640
events = CEp99-tail, margin-tail, curvature-spike, role-signal, FT7-style-event
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole
```

### 必须记录

```text
candidate
dataset
seed
event_id
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
acc_delta
control_rank
real_beats_adamwparallel
real_beats_best_lr
real_beats_random
real_beats_noop
functional_event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR.
$$

At least one mechanism metric:

$$
CEp99_{\text{real}}<CEp99_{\text{AdamWParallel}},
$$

or:

$$
MarginP10_{\text{real}}>MarginP10_{\text{AdamWParallel}},
$$

or:

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_paired_replay_branch_curves.svg
p4_real_vs_strong_controls.svg
p4_horizon_effect_heatmap.svg
p4_kmnist_tail_repair.svg
p4_control_rank_matrix.svg
```

---

## P5：Short-run and full-run functional validation

### 目标

验证 paired replay survivor 是否能在连续训练中保持 task-safe advantage。

### Short-run 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
```

### Full-run 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
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
delta_vs_best_lr
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
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{best LR control}}.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

KMNIST repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Mechanism pass：

At least one:

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
p5_macro_delta_vs_controls.svg
p5_kmnist_repair_matrix.svg
p5_seedwise_win_matrix.svg
p5_task_geometry_pareto.svg
p5_ce_tail_margin_panel.svg
p5_ece_nll_panel.svg
p5_functional_channel_usage_trace.svg
```

---

## P6：AdamW-only full-pass repair 并行线

### 目标

Functional 是核心，但 base 仍未 full-pass。  
P6 不能使用 functional update、teacher、loss change、sampler 或 class weight。它并行修 base，使 functional 判断不被 weak base 污染。

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
p6_adamw_fullpass_gap.svg
p6_kmnist_miss_rows.svg
p6_ce_tail_margin.svg
p6_basis_entropy_vs_gap.svg
```

---

## P7：Robustness and external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean setting 偶然有效。

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
p7_noise_robustness_curve.svg
p7_strong_baseline_pareto.svg
p7_external_ready_scorecard.svg
```

---

## 5. Required artifacts

```text
run_manifest.json
contract_audit_v9222.csv
p0_v9221_boundary_reproduction.csv
p1_i1c_p5_failure_autopsy.csv
p2_base_neutral_interface_factory.csv
p3_functional_actuatability_audit.csv
p4_paired_replay_control_gate.csv
p5_short_full_functional_validation.csv
p6_adamw_only_fullpass_repair.csv
p7_robustness_external_ready.csv
interface_failure_trace_v9222.csv
branch_derivative_trace_v9222.csv
functional_channel_usage_trace_v9222.csv
paired_replay_branch_trace_v9222.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9221_boundary_unstable
F3_p5_failure_unattributed
F4_base_neutral_interface_contract_fail
F5_base_neutral_interface_p4_fail
F6_base_neutral_interface_p5_fail
F7_functional_actuatability_lost
F8_paired_replay_control_equivalent
F9_functional_lr_equivalent
F10_short_run_task_drop
F11_full_run_no_macro_kmnist_gain
F12_adamw_fullpass_fail
F13_strong_baseline_explains_gain
F14_robustness_fail
F15_external_not_ready
F16_fake_or_proxy_violation
F17_artifact_missing
```

---

## 6. Route decision

### Route cases

```text
R1-I1cFailureMechanismIdentified:
  P1 identifies why current I1c breaks P5 base.

R2-BaseNeutralInterfaceP5Pass:
  at least one N1-N5 candidate passes contract/P4/P5 near-pass.

R3-FunctionalActuatabilityRetained:
  P5-qualified interface still has non-AdamW functional actuatability.

R4-StrictInterfacePairedReplayPass:
  RealFunctional beats AdamWParallel / best LR in paired replay.

R5-StrictInterfaceFullFunctionalPass:
  short/full validation shows task-safe macro/KMNIST/geometry/tail gain.

R6-GeometryOnlyFunctional:
  curvature/ECE/NLL improves but task does not.

R7-BaseNeutralKillsActuatability:
  freezing/zero-impact repair restores P5 but destroys functional movement.

R8-InterfaceStillBreaksBase:
  all base-neutral candidates still fail P5 near-pass.

R9-ReturnToBasisFactory:
  current dual-role interface family cannot satisfy P4/P5/function actuatability.

R10-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9221_boundary_pass
i1c_failure_mechanism
best_base_neutral_candidate
interface_contract_pass
interface_p4_pass
interface_p5_nearpass
functional_actuatability_pass
paired_replay_pass
short_run_pass
full_run_pass
functional_task_safe
functional_control_pass
functional_system_pass
functional_kmnist_repair_pass
adamw_fullpass
strong_baseline_pass
robustness_pass
external_ready
primary_blocker
next_required_implementation
success_v9222_strict_purekan_functional
success_v9222_full_functional
success_v9222_external_ready
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.21 boundary。

Step 2:
  P1 做 I1c P5 failure autopsy。
  不知道失败机制前，不做盲目 interface sweep。

Step 3:
  P2 做 base-neutral interface factory。
  优先 N1 zero-impact、N4 role-decoupled、N5 warmstart attach。

Step 4:
  P3 对 P5-qualified candidates 做 functional actuatability audit。
  防止把 functional channel 冻死。

Step 5:
  P4 paired replay control gate。
  必须超过 AdamWParallel 和 best LR。

Step 6:
  P5 short/full functional validation。

Step 7:
  P6 并行 AdamW-only full-pass repair。

Step 8:
  P7 robustness / strong baseline / external-ready gate。
```

---

## 8. 停止条件

### Minimum success

```text
v9.2.21 boundary reproduced
I1c failure mechanism identified
at least one strict interface passes contract/P4/P5 near-pass
functional actuatability retained
paired replay beats AdamWParallel / best LR
no fake/proxy/offload/loss/teacher violation
```

### Full functional success

```text
Minimum success
+
short/full run task-safe mechanism gain
+
KMNIST repair or macro improvement
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
1. v9.2.21 boundary cannot be reproduced；
2. I1c P5 failure cannot be attributed；
3. all base-neutral candidates fail P4；
4. all base-neutral candidates fail P5 near-pass；
5. base-neutral repair kills functional actuatability；
6. all P5-qualified candidates are control-equivalent；
7. full run gives no macro/KMNIST/geometry/tail gain；
8. functional breaks system gate；
9. gains are explained by QuadraticFeatureMLP；
10. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：base-neutral interface P5 pass and actuatability pass

可以声明：

```text
strict PureKAN functional interface now preserves base qualification and can reopen paired replay.
```

但还不能声明 full functional success，除非 P4/P5 full validation 通过。

### Case B：base-neutral P5 pass but actuatability lost

必须声明：

```text
we restored base trainability by silencing the functional channel, but failed to preserve functional interface.
```

下一步回到 interface design，不进入 full functional。

### Case C：paired replay passes

可以声明：

```text
v8-FT7 functional core has been locally transferred into strict FC-PureKAN.
```

但还不能声明 Beyond-MLP，除非 full validation / robustness / strong baseline 通过。

### Case D：all interface candidates still break base

必须声明：

```text
current dual-role interface family is not base-compatible; return to basis / primitive interface factory.
```

### Case E：full run improves KMNIST or macro safely

可以声明：

```text
strict FC-PureKAN functional update has been recovered under strong controls.
```

---

## 10. 最终建议

v9.2.22 的一句话策略是：

$$
\boxed{
\text{先让 strict functional interface 不破坏 base，再证明它保留 non-LR functional actuatability。}
}
$$

当前最关键的问题不是 functional 是否存在，也不是 P4 是否可做，而是：

```text
1. I1c 为什么在 MNIST seed2 / KMNIST seed0 上破 P5？
2. functional channel 是否过早参与 base fitting？
3. branch ratio / derivative scale 是否超出 FT7 安全区间？
4. 是否可以通过 zero-impact / role-decoupled / warmstart attach 让 interface base-neutral？
5. base-neutral 后 functional channel 是否仍有 non-AdamW actuatability？
6. RealFunctional 能否最终超过 AdamWParallel / best LR controls？
```
