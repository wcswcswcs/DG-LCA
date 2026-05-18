# DG-KAN v9.2.21 Strict PureKAN Functional Interface Transfer 完整实验计划

> 本计划基于 v9.2.20 `FunctionalCore FullReplay StrictPureKAN Interface` 的真实结果制定。  
> v9.2.20 已经把 functional update 从 “暂停但不放弃” 推进为 **functional core retained**：v8 functional core 通过了 short-run 与 full replay strong-control 检验。  
> 因此 v9.2.21 的核心不再是继续证明 v8 有没有 functional 信号，而是要把这个信号迁移到 strict FC-PureKAN 内部。
>
> 本轮继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal / margin / calibration loss、no sampler / class weight、no CPU offload、no fake / proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv / PureKANFormer 继续 deferred。**
>
> 本轮的核心问题是：
>
> $$
> \boxed{
> \text{v8-FT7 的 role-wise functional core 如何变成 strict FC-PureKAN 的 edge-owned functional interface？}
> }
> $$
>
> 也就是说，v9.2.21 不再做旧的 output-target patch，也不再调 SNR threshold / event coverage / step fraction。  
> 它要把 v8 中真正有效的 functional 机制抽出来，重建为 PureKAN 内部合法的 edge-function 通道，并且必须在 AdamWParallel / best LR / NoOp / Random / Shuffled controls 下通过 paired replay 与 full validation。

---

## 0. 当前实验结果的独立判断

### 0.1 v9.2.20 达成了什么

v9.2.20 的 terminal route 是：

```text
route = R2-v8FunctionalFullReplayPass
base_candidate = LQ-t2-h256
success_v9220_functional_core_retained = true
success_v9220_strict_purekan_functional = false
success_v9220_external_ready = false
```

这说明 functional update 的核心方向已经被重新保留下来。v9.2.20 不是 one-step source recap，而是完成了：

```text
P1:
  v8 50/240-step strong-control replay；
  v8_short_run_pass = 1；
  row survivor count = 50。

P2:
  640-step full replay；
  v8_full_replay_pass = 1；
  v8_full_replay_survivor_count = 2；
  functional row survivor count = 22/30。
```

其中 `V8-FT7-RoleWiseFunctional` 是最重要的 functional core 证据。它在 full replay 中同时保持：

```text
beats_adamwparallel_rate = 1.0
beats_best_lr_rate = 1.0
```

并且 curvature delta 与 AdamWParallel / LR controls 明显不同。`V8-Adaptive-FT-P` 也有 row-level survivor，但 aggregate control beat rate 是 `0.866667`，机制不如 FT7 干净。`V8-ShuffledRoleMask` 的 CEp99 / curvature 很接近 FT7，但作为 control branch 没有通过 beats AdamWParallel / best LR gate，这说明 role-wise functional 的信号不是简单随机 role masking 可以完全解释。

### 0.2 v9.2.20 没有达成什么

v9.2.20 仍然没有达成 strict PureKAN functional success。下游边界非常明确：

```text
p4_strict_purekan_functional_interface_design.csv:
  not_run

p5_basis_factory_functional_actuatability.csv:
  not_run

p6_adamw_only_fullpass_repair.csv:
  not_run

p7_functional_survivor_full_validation.csv:
  not_run

p8_robustness_external_ready.csv:
  not_run
```

因此当前不能声明：

```text
strict FC-PureKAN functional update 已成功；
LQ/A7c 已经恢复 functional advantage；
external fair 可以打开；
PureKANConv / PureKANFormer 可以打开。
```

### 0.3 当前最重要的判断

v9.2.20 把问题从：

```text
functional update 是否真的有核心信号？
```

推进为：

```text
如何把 v8 functional core 迁移成 strict FC-PureKAN interface？
```

也就是说，当前 blocker 已经不是 functional 本身，也不是 v8 证据不足，而是：

$$
\boxed{
\text{strict PureKAN functional interface 尚未实现。}
}
$$

---

## 1. 本轮总体目标

v9.2.21 的总体目标是：

$$
\boxed{
\text{把 v8-FT7 role-wise functional core 迁移成 strict FC-PureKAN edge-owned functional interface。}
}
$$

这不是简单把 v8 stack 搬进 v9。v8 的 `linear-SiLU stack + KAN-style head` 是 transitional architecture，不能作为终极 PureKAN。我们要迁移的是机制，而不是架构外壳。

本轮必须回答六个问题。

### Q1：FT7 的成功机制到底是什么

必须用 v9.2.20 的 full replay survivors 反推出 FT7 的机制：

```text
role-wise update 是否必要？
stack/head role 分工如何？
functional update 是否与 AdamW 不完全平行？
branch ratio 与 CE-tail / margin / curvature gain 是否相关？
effective derivative scale 是否在安全区间内？
ShuffledRoleMask 为什么接近但不过 gate？
```

### Q2：strict FC-PureKAN 中对应的 role 是什么

v8 的 roles 不能直接复制到 v9。必须映射成 PureKAN edge-system 内部 roles：

```text
lift_identity role
task_T2 role
functional_channel role
output_linear role
basis_scale / derivative_scale role
actuator role
```

### Q3：strict PureKAN 是否需要 dual-role edge basis

当前 LQ/A7c 的失败说明：可控 actuator 不等于 functional interface。  
v9.2.21 要验证一种更结构化的接口：

$$
\phi_{jc}(h)
=
\phi^{task}_{jc}(h)
+
\phi^{func}_{jc}(h).
$$

其中 $\phi^{task}$ 承担 AdamW base task fitting，$\phi^{func}$ 只在 functional event 中承担 role-wise correction。二者都必须是 edge function 内部 channel，不允许 external residual。

### Q4：functional 是否仍能超过 AdamWParallel / LR controls

迁移后的 strict PureKAN functional 必须满足：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

否则它只是 LR / extra-step disguise。

### Q5：base 是否仍然 near-pass / full-pass

Functional 是核心，但不能破坏 base。任何 strict PureKAN interface candidate 必须先满足：

```text
P4 pass；
P5 near-pass；
no teacher / no loss / no external residual；
manual forward/backward/update。
```

### Q6：何时才能打开 external-ready / Conv / Former

只有 strict PureKAN functional route 同时满足：

```text
P4 pass
P5 near-pass or full-pass
paired replay control pass
short/full validation pass
robustness pass
strong baseline pass
```

才允许进入 external-ready。PureKANConv / PureKANFormer 继续 deferred。

---

## 2. 核心假设

### H1：FT7 的关键不是 output target，而是 role-wise safe branch interface

v9.2.9-v9.2.17 的 output-target / actuator / signal-metric 路线失败，v9.2.20 的 FT7 full replay 成功，说明真正有效的 functional 机制不是“手写一个 output-space correction”，而是 role-wise branch interface。

H1 成立标准：

在 P1 mechanism extraction 中，FT7 满足：

```text
role update 分布与 gain 有显著关联；
functional 与 AdamW 的 cosine 不总是 > 0.95；
branch ratio 与 CEp99 / margin / curvature gain 有正相关；
ShuffledRoleMask 接近但不能稳定 beats AdamWParallel / best LR。
```

至少要求：

$$
Corr(r_{\text{branch}}, -\Delta CEp99)\geq0.30
$$

或：

$$
Corr(r_{\text{branch}}, \Delta MarginP10)\geq0.30
$$

或：

$$
Corr(r_{\text{branch}}, -\Delta Curvature)\geq0.30.
$$

### H2：strict PureKAN 需要 task channel 与 functional channel 分离

当前 LQ/A7c 同一组参数既承担 task fitting，又承担 functional correction，容易退化成 AdamWParallel 或被 projection 中和。Dual-role edge basis 通过将 task channel 和 functional channel 分开，提供更安全的 non-LR actuatability。

H2 成立标准：

dual-role candidate 相比 single-channel LQ/A7c 满足：

$$
r_{z,\perp}\geq0.05,
$$

其中 $r_{z,\perp}$ 是去除 AdamW-parallel 成分后的 output displacement ratio，并且：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

### H3：functional channel 必须有 derivative-scale controller

v8 中 effective derivative scale 可能是 functional safety 的关键。strict PureKAN functional channel 需要控制：

$$
s_{\text{eff}}
=
|\alpha|\cdot B'_a(h)_{p95}.
$$

如果 $s_{\text{eff}}$ 太小，functional 被中和；太大，则伤 task。H3 要验证安全区间。

H3 成立标准：

存在区间：

$$
s_{\min}\leq s_{\text{eff}}\leq s_{\max}
$$

使得：

```text
bad_event_rate <= 0.05
paired replay beats controls
task safety pass
```

### H4：bounded rational / piecewise / shared RBF 更适合作为 functional channel，而 T2 更适合作为 task channel

T2 已经证明是当前最好的 task basis，但 functional channel 可能需要更局部或 bounded 的形状。候选 functional basis 包括：

```text
bounded rational
piecewise linear 2/4
shared RBF4
centered/normalized T2
```

H4 成立标准：

至少一个非 T2 functional channel 在 paired replay 中超过 T2-only functional channel，同时保持 P4/P5。

### H5：如果 strict interface 失败，下一步应回到 basis/primitive factory，而不是再调 target/gate

如果所有 strict interface candidates 都不能超过 AdamWParallel / LR controls，则说明当前 interface 设计仍未捕捉 v8 机制。此时下一步应回到 primitive / basis factory，而不是继续 patch output target。

---

## 3. Candidate 设计

### 3.1 Baselines

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current strict FC-PureKAN base。

A7c-BasisEntropy-ValueOnly:
  P4-qualified actuator reference。

V8-FT7-RoleWiseFunctional:
  mechanism source, not final strict candidate。
```

### 3.2 Strict PureKAN interface candidates

#### I1：Dual-role T2 + bounded rational functional channel

$$
\phi_{jc}(h)
=
c^{task}_{jc,0}h
+
c^{task}_{jc,2}T_2(h)
+
c^{func}_{jc,r}
\frac{h^2}{1+\beta h^2}.
$$

其中 $\beta$ 可以固定或 edge-owned，但第一轮优先固定，避免 backward overhead 和 shape instability。

Candidate IDs：

```text
I1a-T2Task-RationalFunc-FixedBeta
I1b-T2Task-RationalFunc-ValueOnly
I1c-T2Task-RationalFunc-DerivativeControlled
```

#### I2：Dual-role T2 + piecewise local functional channel

$$
B_a(h)=\operatorname{clip}(h-\tau_1,0,\tau_2-\tau_1)
$$

或 two-bin hat basis。它提供局部 correction，适合 CE-tail / margin-tail。

Candidate IDs：

```text
I2a-T2Task-Piecewise2Func-FixedKnots
I2b-T2Task-Piecewise4Func-FixedKnots
I2c-T2Task-PiecewiseFunc-DerivativeControlled
```

#### I3：Dual-role T2 + shared RBF functional channel

$$
B_a(h)=\exp\left(-\frac{(h-\mu_a)^2}{2\sigma^2}\right).
$$

只允许 shared centers，避免 dense center materialization。

Candidate IDs：

```text
I3a-T2Task-SharedRBF4Func-FixedCenters
I3b-T2Task-SharedRBF4Func-ValueOnly
```

#### I4：Role-wise PureKAN metric

Functional 不新增 output target，只对 AdamW update 施加 role-wise metric：

$$
\theta_{t+1}
=
\theta_t+
P_{\text{role}}(\theta,t)\Delta\theta_{\text{AdamW}}.
$$

其中：

$$
P_{\text{role}}
=
I+\sum_r \alpha_r P_r.
$$

候选：

```text
I4a-RoleSNRMetric
I4b-RoleCurvatureDampedMetric
I4c-FT7StyleRoleGuard
I4d-TailEventRoleMetric
```

必须证明不是 scalar LR disguise。

#### I5：Orthogonal non-AdamW functional correction

构造功能通道中的非 AdamW 平行分量：

$$
\Delta\theta_{\perp}
=
\Delta\theta_{\text{func}}
-
\frac{
\langle \Delta\theta_{\text{func}},\Delta\theta_{\text{AdamW}}\rangle
}{
\|\Delta\theta_{\text{AdamW}}\|^2+\epsilon
}
\Delta\theta_{\text{AdamW}}.
$$

候选：

```text
I5a-RationalFunc-OrthogonalCorrection
I5b-PiecewiseFunc-OrthogonalCorrection
I5c-SharedRBFFunc-OrthogonalCorrection
```

### 3.3 Controls

每个 candidate 必须配套 controls：

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
ShuffledFuncChannel
FrozenFuncChannel
```

### 3.4 禁止项

任何 candidate 如果出现以下情况，直接 contract fail：

```text
external_residual_used = 1
ordinary_mlp_path_used = 1
non_edge_owned_param = 1
teacher_used = 1
loss_modified = 1
sampler_or_class_weight_changed = 1
uses_loss_backward = 1
fake_or_proxy = 1
```

---

## 4. 实验阶段

## P0：v9.2.20 boundary reproduction

### 目标

复现 v9.2.20 的 terminal boundary，确认本轮建立在真实 functional core retained 之上。

### 必须记录

```text
route
v8_short_run_pass
v8_short_run_row_pass_count
v8_full_replay_pass
v8_full_replay_survivor_count
functional_row_survivor_count
FT7_row_survivor_count
Adaptive_row_survivor_count
beats_adamwparallel_rate
beats_best_lr_rate
strict_purekan_interface_status
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R2-v8FunctionalFullReplayPass
v8_short_run_pass = 1
v8_full_replay_pass = 1
functional_core_retained = 1
strict_purekan_interface_not_implemented = 1
fake_proxy_count = 0
```

### 可视化

```text
p0_v9220_boundary_dashboard.svg
p0_v8_survivor_matrix.svg
p0_ft7_vs_controls_full_replay.svg
```

---

## P1：FT7 mechanism extraction

### 目标

从 v8 full replay survivor 中抽取可迁移机制，避免 blind architecture copying。

### 必须记录

```text
candidate
dataset
seed
event_id
role
branch
steps_or_epoch
role_update_norm
role_snr
role_curvature
role_event_frequency
branch_ratio
effective_derivative_scale
cos_functional_adamw
cos_functional_best_lr
cos_functional_random
functional_norm_vs_adamw
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
control_rank
beats_adamwparallel
beats_best_lr
```

### 关键指标

Branch ratio：

$$
r_{\text{branch}}
=
\frac{\|\alpha F(h)\|}{\|h\|+\epsilon}.
$$

Effective derivative scale：

$$
s_{\text{eff}}
=
|\alpha|\cdot B'_{p95}.
$$

AdamW alignment：

$$
\cos_{\text{AdamW}}
=
\frac{
\langle \Delta\theta_{\text{functional}},\Delta\theta_{\text{AdamW}}\rangle
}{
\|\Delta\theta_{\text{functional}}\|\|\Delta\theta_{\text{AdamW}}\|+\epsilon
}.
$$

Non-LR displacement ratio：

$$
r_{\perp}
=
\frac{
\|\Delta\theta_{\text{functional}}-\operatorname{proj}_{\Delta\theta_{\text{AdamW}}}\Delta\theta_{\text{functional}}\|
}{
\|\Delta\theta_{\text{AdamW}}\|+\epsilon
}.
$$

### 判断标准

FT7 mechanism identified if：

```text
functional beats AdamWParallel and best LR in full replay；
non-trivial role-specific update pattern exists；
r_perp >= 0.01 for survivor events or role-level non-scalar metric explains gain；
branch_ratio/effective_derivative_scale correlates with at least one mechanism gain；
ShuffledRoleMask does not fully explain all survivor metrics。
```

量化要求至少一个成立：

$$
Corr(r_{\text{branch}}, -\Delta CEp99)\geq0.30,
$$

$$
Corr(r_{\text{branch}}, \Delta MarginP10)\geq0.30,
$$

$$
Corr(s_{\text{eff}}, -\Delta Curvature)\geq0.30.
$$

### 可视化

```text
p1_rolewise_mechanism_heatmap.svg
p1_branch_ratio_vs_gain.svg
p1_effective_derivative_scale_vs_curvature.svg
p1_cosine_with_adamw_lr_random.svg
p1_shuffled_role_comparison.svg
```

---

## P2：Strict PureKAN interface contract and gradcheck

### 目标

实现 I1-I5 interface candidates，并验证它们是 strict PureKAN，而不是 transitional stack 或 ordinary residual。

### 必须记录

```text
candidate
interface_family
basis_formula
task_channel_basis
functional_channel_basis
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
non_edge_owned_params
manual_forward
manual_backward
manual_update
uses_loss_backward
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
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

### 可视化

```text
p2_interface_contract_heatmap.svg
p2_gradcheck_by_interface.svg
p2_basis_conditioning_by_interface.svg
p2_functional_channel_entropy.svg
```

---

## P3：P4 / P5 base qualification

### 目标

确保 strict PureKAN interface 不破坏系统 gate 和 AdamW-only trainability。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
baseline = MLP-match
functional_update = off
```

### 必须记录

P4：

```text
forward_ratio_q50
forward_ratio_q90
backward_ratio_q50
backward_ratio_q90
step_ratio_q50
step_ratio_q90
compact_memory_ratio
conservative_memory_ratio
kernel_count
functional_channel_extra_time
```

P5：

```text
dataset
seed
KAN_acc
MLP_match_acc
delta_vs_mlp
near_pass
full_pass
CEp99
margin_p10
ECE
NLL
basis_usage_entropy
functional_channel_usage_entropy
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
compact\_memory\leq1.05.
$$

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Full-pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

### 可视化

```text
p3_task_system_pareto.svg
p3_p4_timing_breakdown.svg
p3_p5_delta_vs_mlp.svg
p3_kmnist_miss_matrix.svg
```

---

## P4：Paired replay control gate for strict interface

### 目标

验证 strict PureKAN functional interface 是否真正超过 AdamWParallel / LR controls。  
这是本轮最核心 gate。

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

Strict interface paired replay pass：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

并且至少一个机制指标同时超过 AdamWParallel 和 best LR：

$$
CEp99_{\text{real}}<CEp99_{\text{AdamWParallel}},
$$

$$
CEp99_{\text{real}}<CEp99_{\text{best LR}},
$$

或：

$$
MarginP10_{\text{real}}>MarginP10_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{real}}>MarginP10_{\text{best LR}},
$$

或：

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

Control pass：

```text
RealFunctional beats Random and NoOp on at least one mechanism metric；
ShuffledRoleMask cannot reproduce all gains；
InvertedRole must not pass.
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
p4_paired_replay_branch_curves.svg
p4_real_vs_strong_controls.svg
p4_horizon_effect_heatmap.svg
p4_control_rank_matrix.svg
p4_kmnist_tail_repair.svg
```

---

## P5：Short-run and full-run functional validation

### 目标

只有 P4 survivor 才能进入 P5。验证 strict PureKAN functional interface 能否在连续训练中保持优势。

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

至少一个：

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

Functional 是核心，但 base 仍未 full-pass。并行推进 AdamW-only full-pass，避免 functional 掩盖 base gap。

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

### Forbidden

```text
teacher
distillation
label smoothing
loss change
sampler
class weight
functional update
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

## P7：Robustness and strong baseline gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family 偶然有效。

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
contract_audit_v9221.csv
p0_v9220_boundary_reproduction.csv
p1_ft7_mechanism_extraction.csv
p2_strict_purekan_interface_contract_gradcheck.csv
p3_interface_p4_p5_base_qualification.csv
p4_strict_interface_paired_replay_controls.csv
p5_strict_interface_short_full_validation.csv
p6_adamw_only_fullpass_repair.csv
p7_robustness_strong_baseline_gate.csv
role_mechanism_trace_v9221.csv
purekan_interface_trace_v9221.csv
paired_replay_branch_trace_v9221.csv
functional_channel_usage_trace_v9221.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9220_boundary_unstable
F3_ft7_mechanism_unattributed
F4_interface_contract_fail
F5_interface_grad_fail
F6_interface_p4_fail
F7_interface_p5_nearpass_fail
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
R1-FT7MechanismIdentified:
  v8 FT7 role-wise mechanism is extracted and explains full replay survivor.

R2-DualRoleInterfaceContractPass:
  strict PureKAN dual-role edge interface passes contract / grad / P4 / P5 near-pass.

R3-StrictPureKANFunctionalPairedPass:
  strict interface RealFunctional beats AdamWParallel / best LR controls in paired replay.

R4-StrictPureKANFunctionalFullPass:
  short/full run shows task-safe macro/KMNIST/geometry/tail gain.

R5-GeometryOnlyFunctional:
  curvature/ECE/NLL improves but task does not.

R6-InterfaceControlEquivalent:
  strict interface cannot beat strong controls.

R7-InterfaceBreaksBase:
  functional channel breaks P4 or P5 near-pass.

R8-ReturnToBasisFactory:
  FT7 mechanism valid but current interface candidates fail; continue primitive/basis redesign.

R9-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R10-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9220_boundary_pass
ft7_mechanism_identified
best_interface_candidate
interface_contract_pass
interface_p4_pass
interface_p5_nearpass
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
success_v9221_strict_purekan_functional
success_v9221_full_functional
success_v9221_external_ready
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.20 boundary，确认 functional_core_retained = true。

Step 2:
  P1 做 FT7 mechanism extraction。
  不知道机制前，不允许盲目设计 strict interface。

Step 3:
  P2 实现 I1-I5 strict PureKAN interface candidates，并做 contract / gradcheck / interaction audit。

Step 4:
  P3 做 P4/P5 base qualification。
  没有 P4/P5 candidate，不进入 paired replay。

Step 5:
  P4 做 paired replay control gate。
  必须超过 AdamWParallel 和 best LR。

Step 6:
  P5 做 short-run 和 full-run validation。

Step 7:
  P6 并行 AdamW-only full-pass repair。

Step 8:
  P7 做 robustness / strong baseline / external-ready gate。
```

---

## 8. 停止条件

### Minimum success

```text
v9.2.20 boundary reproduced
FT7 mechanism identified
at least one strict PureKAN interface passes contract + grad + P4 + P5 near-pass
paired replay beats AdamWParallel / best LR controls
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
1. v9.2.20 boundary cannot be reproduced；
2. FT7 mechanism cannot be identified；
3. all strict interface candidates violate PureKAN contract；
4. all strict interface candidates fail P4 or P5 near-pass；
5. all strict interface candidates are control-equivalent；
6. full run gives no macro/KMNIST/geometry/tail gain；
7. functional breaks system gate；
8. gains are explained by QuadraticFeatureMLP；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：strict interface paired replay passes

可以声明：

```text
v8 functional core has been transferred into strict FC-PureKAN at local causal level.
```

但还不能声明 Beyond-MLP，除非 P5/P7 也通过。

### Case B：strict interface full run passes

可以声明：

```text
strict FC-PureKAN functional update has been recovered under strong controls.
```

### Case C：FT7 mechanism valid but interface fails

必须声明：

```text
functional concept remains valid, but current strict PureKAN interface candidates cannot yet carry it.
```

下一步回到 basis / primitive interface factory。

### Case D：base full-pass but functional fails

必须声明：

```text
strict FC-PureKAN base improved; functional advantage remains unproven in current interface.
```

### Case E：all candidates control-equivalent

必须声明：

```text
v8 functional core does not automatically transfer to the tested strict PureKAN interfaces; current interface family must be redesigned, not patched.
```

---

## 10. 最终建议

v9.2.21 的一句话策略是：

$$
\boxed{
\text{把 v8-FT7 的 role-wise functional core 迁移成 strict FC-PureKAN 内部的 dual-role edge functional interface。}
}
$$

当前最重要的问题不是 functional 是否存在，也不是继续调 output target，而是：

```text
1. FT7 full replay survivor 的 role-wise 机制是什么？
2. ShuffledRoleMask 为什么接近但不过 gate？
3. strict FC-PureKAN 中如何定义 task channel 与 functional channel？
4. functional channel 如何保持 edge-owned、P4-qualified、P5 near-pass？
5. RealFunctional 能否超过 AdamWParallel / best LR controls？
6. 它能否修复 KMNIST、CE-tail、margin-tail，并最终进入 external-ready？
```
