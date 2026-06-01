# DG-KAN v9.2.13 Functional Controllability 与 PureKAN Actuator Redesign 完整实验计划

> 本计划基于 v9.2.12 `Functional Direction Reconstruction` 的真实 terminal route 制定。  
> v9.2.12 的结论非常关键：**不是 LQ base 失效，不是 SNR overhead 失效，不是 event coverage 失效，而是当前 LQ 参数子空间无法产生足够强、足够可控、足够超过 controls 的 output-space functional effect。**  
> 因此 v9.2.13 不继续给旧 functional direction 加 gate，也不继续调 threshold / stride / step fraction。  
> v9.2.13 的核心目标是回答：
>
> $$
> \boxed{
> \text{当前 FC-PureKAN LQ 是否具备 functional controllability？如果不具备，怎样在 strict PureKAN 内重设计可控 actuator？}
> }
> $$
>
> 这里的 actuator 不是外部 residual，不是普通 MLP hidden path，不是 Conv / Former，不是 teacher，不是 loss。它必须仍然是 FC-PureKAN edge-basis 系统中的合法参数子空间，只是专门为 functional update 提供可控、低秩、可审计的函数空间移动能力。

---

## 0. 当前状态与独立判断

### 0.1 v9.2.12 的真实 terminal route

v9.2.12 执行到：

```text
route = R5-ControlEquivalentAgain
base_candidate = LQ-t2-h256
success_v9212_output_direction = false
success_v9212_event_causality = false
success_v9212_full_functional = false
success_v9212_external_ready = false
```

P0 复现 v9.2.11 边界，source route 仍为 `R5-FunctionalControlEquivalent`，source P2 causality pass count 为 `0`。P1 autopsy 成立，旧 functional 的主要失败机制是：

```text
event degeneracy
control-equivalence
output-effect too small
projection neutralization
```

P2 output-space direction factory 真实执行：

```text
rows = 116640
base = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
priority combos = 12 output-target / subspace / solver
events = E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent
branches = 6
horizons = 1,5,20
```

但没有任何 output-space direction 通过 gate：

```text
p2_output_direction_pass_count = 0
```

这次有一个很重要的正向修复：v9.2.11 的 event degeneracy 已被修复，group coverage 进入计划范围：

```text
coverage = 0.033333 or 0.100000
```

但 output-space directions 仍然弱：

```text
max group mean output displacement ratio = 0.036009 < 0.05
max output target fit R2 = 0.059320
```

因此，P4 paired replay causal confirmation、P5 short-run、P6 full re-entry、P7 robustness、P8 external-ready 全部正确保持 `not_run`。

### 0.2 目标是否达成

没有。

当前已经达成：

```text
1. Clean FC-PureKAN LQ base 成立；
2. LQ P4 system gate 成立；
3. LQ P5 robust near-pass 成立；
4. low-cost SNR instrumentation 可运行；
5. one-step safety 可以通过 D9 修复；
6. paired replay infrastructure 已跑通；
7. sparse event coverage 已从退化状态修复到计划范围。
```

当前未达成：

```text
1. output-space direction pass；
2. event-level causal advantage；
3. short-run functional advantage；
4. full 10-seed functional re-entry；
5. robustness / noise advantage；
6. strong baseline / external-ready；
7. significant Beyond-MLP。
```

因此不能声明：

```text
functional update 成功；
functional update 是 DG-KAN 的核心贡献；
PureKAN 已显著超过 MLP；
可以进入 external fair；
可以打开 PureKANConv / PureKANFormer。
```

### 0.3 最重要的独立判断

v9.2.12 的关键不是 “再失败一次”，而是把 functional 主线推进到一个更根本的问题：

$$
\boxed{
\text{当前 LQ-t2-h256 参数子空间对想要的 output-space target 不可控或弱可控。}
}
$$

证据是：

```text
event coverage 已修好；
target / subspace / solver 组合真实测了 116640 rows；
但 target fit R2 最大只有 0.059320；
output displacement ratio 最大只有 0.036009；
没有一个 group 过 output-direction gate。
```

这说明：继续给旧方向加 SNR gate、task-safe projection、event controller，很可能只是让 update 更安全，但不会让它更有用。当前问题不是 “什么时候触发”，而是 “触发后能不能真的移动到有价值的函数方向”。

---

## 1. v9.2.13 的整体目标

v9.2.13 的整体目标是：

$$
\boxed{
\text{建立 functional controllability audit，并在 strict FC-PureKAN 内设计可控 functional actuator。}
}
$$

本轮不以 final accuracy 为第一目标，而是要回答四个问题。

### Q1：当前 LQ 子空间到底能不能实现 output-space target？

对每个 target $t=\Delta z_{\text{target}}$ 和参数子空间 $U$，定义：

$$
\delta_U^\star
=
\arg\min_{\delta\in U}
\|J_U\delta-t\|_2^2+\lambda\|\delta\|_2^2.
$$

定义 target fit：

$$
R^2_{\text{fit}}
=
1-
\frac{\|J_U\delta_U^\star-t\|_2^2}
{\|t\|_2^2+\epsilon}.
$$

定义输出位移强度：

$$
r_z
=
\frac{\|J_U\delta_U^\star\|_2}
{\|\Delta z_{\text{AdamW}}\|_2+\epsilon}.
$$

v9.2.12 中：

```text
max R2_fit = 0.059320
max r_z = 0.036009
```

v9.2.13 需要判断这是：

```text
target 本身不合理；
solver 过弱；
task-safe projection 过度中和；
还是 LQ 参数子空间没有足够 controllability。
```

### Q2：是否存在 direct logit oracle target 本身有用？

在不更新参数的 diagnostic 中，直接对 logits 施加小的 $\Delta z_{\text{target}}$，观察 CE tail / margin / ECE / NLL 是否改善。

如果 direct logit target 本身无效，说明 output target 错。  
如果 direct logit target 有效，但参数子空间无法拟合，说明 actuator 不足。

### Q3：如何在 strict FC-PureKAN 内增加 functional actuator？

Functional actuator 必须仍然是 edge-owned basis channel。允许：

```text
additional normalized T2 channel
bounded rational channel
piecewise-linear local channel
tail-control output edge-basis channel
basis-entropy balancing channel
low-rank edge coefficient actuator
```

不允许：

```text
external residual
ordinary Linear shortcut
MLP hidden path
trainable Conv / preprocessing
teacher / loss modification
class weight / sampler
```

### Q4：actuator 是否能同时保留 AdamW near-pass 与 functional controllability？

Actuator candidate 不能只让 functional 好用却破坏 base。它必须先满足：

```text
PureKAN equivalence pass
GradPass
P4 system pass
P5 near-pass
```

然后才允许重新打开 functional direction / paired replay。

---

## 2. 本轮硬约束

### 2.1 Clean training contract

所有 official candidate 必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
distillation_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

训练目标保持：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

Functional update 是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\lambda_f\Delta\theta_{\text{functional}}.
$$

不允许：

$$
L=CE+\lambda L_{\text{geo}}.
$$

### 2.2 FC-PureKAN actuator contract

LQ 基础结构必须保持 FullEdge composition：

$$
h_j=\sum_i c^0_{ij}B_0(x_i),
$$

其中：

$$
B_0(x)=x.
$$

输出层必须仍然是 edge-basis layer：

$$
y_c
=
\sum_j
\sum_k
c^1_{jc,k}B_k(h_j).
$$

如果加入 functional actuator basis，则形式为：

$$
y_c
=
\sum_j
\left(
c^1_{jc,0}B_0(h_j)
+
c^1_{jc,2}B_2(h_j)
+
c^1_{jc,a}B_a(h_j)
\right),
$$

其中 $B_a$ 必须是合法 edge basis，例如 normalized T2、bounded rational、piecewise-linear、local RBF-like shared center、或 basis-entropy actuator。  
所有新参数必须属于 $c^1_{jc,a}$ 或 edge-owned source / basis parameters。

### 2.3 Functional 不得替代 AdamW base

P5 AdamW-only near-pass 必须单独报告。Functional branch 不能用来掩盖 AdamW base 失败。

### 2.4 Conv / Former deferred

继续记录：

```text
PureKANConv_status = deferred_until_FC_PureKAN_functional_advantage_or_user_unlock
PureKANFormer_status = deferred_until_FC_PureKAN_functional_advantage_or_user_unlock
```

不产生 Conv / Former measured candidate。

---

## 3. 核心假设

### H1：v9.2.12 失败主要来自 functional controllability 不足，而不是 event controller

因为 v9.2.12 中 event coverage 已经进入 $0.033333$ 或 $0.100000$，不再是 v9.2.11 的 E0/E2 全开或 E5 全关。因此，本轮优先假设 event 不是主因。

H1 成立标准：

若当前 LQ 子空间在 P1 中满足：

$$
R^2_{\text{fit}}<0.10,
$$

且：

$$
r_z<0.05,
$$

而 direct logit oracle target 在 P2 中能改善 CE tail / margin，则 H1 成立。

### H2：output target 本身可能有效，但当前参数子空间无法实现

H2 需要通过 direct logit oracle 验证。若直接施加 $\Delta z_{\text{target}}$ 可以改善：

$$
CEp99,
$$

或：

$$
MarginP10,
$$

或：

$$
ECE / NLL,
$$

但参数求解 $J_U\delta$ 的 fit R2 很低，则说明需要 actuator redesign，而不是 target redesign。

### H3：LQ-T2 的二阶 basis 足够做 task near-pass，但不足以给 functional update 提供独立控制通道

LQ-T2 的成功来自低阶 interaction；但同一组参数既负责 base task fit，又负责 functional correction，functional update 会被 task-safe projection 消掉。  
因此需要分离：

```text
task basis channel
functional actuator basis channel
```

但必须保持 strict PureKAN。

H3 成立标准：

加入 actuator 后，candidate 满足：

$$
R^2_{\text{fit,new}}\geq R^2_{\text{fit,LQ}}+0.10,
$$

或：

$$
r_{z,new}\geq0.05,
$$

且 P4/P5 不退化。

### H4：如果 actuator 通过 controllability 但不能产生 paired replay causality，则 output target 错

如果 actuator 使 $R^2_{\text{fit}}$ 提高到至少 $0.20$，且 $r_z\geq0.05$，但 paired replay 仍 control-equivalent，则说明问题转向 output target / mechanism，不是 actuator。

### H5：如果 actuator 破坏 P5 near-pass，则它不是可用 functional base

Functional actuator 必须是 task-safe base 的一部分。若加入 actuator 后：

$$
\Delta Acc_{\text{macro}}<-0.01
$$

或 near-pass rate < 0.80，则不得进入 functional branch。

---

## 4. Candidate 设计

### 4.1 Base references

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current FC-PureKAN near-pass base。

LQ1-LQ-t2-h256-fanin-output-scale:
  v9.2.7 best repair reference。
```

### 4.2 Existing subspaces for audit

```text
S1-QuadraticCoeffSubspace:
  current T2 coefficient subspace。

S2-LiftSubspace:
  identity lift coefficient subspace。

S3-OutputLinearSubspace:
  output linear coefficient subspace。

S4-LiftPlusQuadraticSubspace:
  lift + T2 coefficients。

S5-RecentSignalSubspace:
  coherent gradient top-k subspace。

S6-OrthogonalToAdamWSubspace:
  remove AdamW-parallel component。
```

### 4.3 Actuator candidates

所有 actuator 都必须是 FC-PureKAN edge-owned。

```text
A0-LQ-current:
  reference。

A1-LQ-T2ActuatorZeroInit:
  额外 T2 actuator channel，zero / tiny init，只作为可控子空间。

A2-LQ-NormalizedT2Actuator:
  normalized T2 actuator，控制 lifted coordinate scale。

A3-LQ-CenteredT2Actuator:
  centered T2 actuator，减少 dominant basis。

A4-LQ-BoundedRationalActuator:
  bounded rational basis:
  B_a(x)=x^2/(1+\beta x^2)，避免 tail explosion。

A5-LQ-PiecewiseLinear2Actuator:
  two-bin active piecewise-linear basis，提供 local tail correction。

A6-LQ-LocalRBFSharedCenterActuator:
  shared-center local bump basis，中心固定或 edge-owned但低秩共享。

A7-LQ-BasisEntropyActuator:
  专门调节 basis usage entropy 的合法 edge-basis channel。

A8-LQ-LowRankOutputActuator:
  低秩 output-edge coefficient actuator：
  C_{j,c,a}=\sum_r u_{j,r}v_{c,r}a_r。
```

### 4.4 Output targets

```text
O1-HardTailLogitCorrection:
  减少 CEp99 / wrong confidence tail。

O2-MarginTailExpansion:
  提升 margin_p10。

O3-CalibrationTailCompression:
  压缩 wrong-confidence logit norm。

O4-CurvatureOutputFlattening:
  降低 local curvature / local Lipschitz proxy。

O5-BasisEntropyOutputCorrection:
  让输出对 basis channels 的敏感性更均衡。

O6-KMNISTHardModeOutputTarget:
  针对 KMNIST hard-mode 的 output-space target，不使用 class weight / sampler。
```

### 4.5 Solvers

```text
SOL0-ProjectedGradient:
  delta = J^T target，然后 task-safe projection。

SOL1-LeastSquaresSketch:
  low-rank sketch solve min ||J delta - target||^2。

SOL2-ConstrainedLeastSquares:
  加 holdout non-increase constraint。

SOL3-TrustRegionQP:
  限制 ||delta|| <= rho ||AdamW step||。

SOL4-AbstainingSolver:
  fit / safety confidence 低时 skip。
```

### 4.6 Controls

```text
C0-AdamWOnly
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-ShuffledTarget
C4-ShuffledSNRMask
C5-AdamWParallelDirection
C6-InvertedEventController
```

---

## 5. 实验阶段

## P0：复现 v9.2.12 terminal boundary

### 目标

确认当前不是测量噪声，并复现 control-equivalent / weak output effect 结论。

### 必须记录

```text
source_route
p2_output_direction_pass_count
max_output_displacement_ratio
max_output_target_fit_R2
event_coverage_min
event_coverage_max
branch_count
row_count
fake_proxy_count
```

### 判断标准

复现通过：

```text
p2_output_direction_pass_count = 0
max_output_displacement_ratio < 0.05 ± tolerance
max_output_target_fit_R2 < 0.10
event coverage in [0.03,0.15]
```

### 可视化

```text
p0_v9212_reproduction_dashboard.svg
p0_target_fit_and_output_ratio.svg
p0_event_coverage_check.svg
```

---

## P1：Functional controllability audit

### 目标

量化当前 LQ 参数子空间对 output targets 的可控性。P1 是本轮第一核心阶段。

### 方法

对每个 target $O_i$、subspace $S_j$、solver $SOL_k$，在同一 checkpoint 上计算：

$$
\delta^\star
=
\arg\min_{\delta\in S_j}
\|J_{S_j}\delta-\Delta z_{O_i}\|^2+\lambda\|\delta\|^2.
$$

同时计算 task-safe projection 后的有效位移。

### 必须记录

```text
candidate
dataset
seed
target_id
subspace_id
solver_id
target_norm
delta_norm
delta_norm_after_projection
norm_after_projection_ratio
output_displacement_norm
output_displacement_ratio_vs_adamw
target_fit_R2
target_residual_norm
cos_delta_adamw
cos_delta_task_grad
holdout_delta
bad_event
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
```

### 判断标准

Controllability weak：

$$
R^2_{\text{fit}}<0.10
$$

and:

$$
r_z<0.05.
$$

Controllability promising：

$$
R^2_{\text{fit}}\geq0.20
$$

and:

$$
r_z\geq0.05.
$$

Projection neutralization：

$$
\frac{\|\delta_{\text{after projection}}\|}
{\|\delta_{\text{before projection}}\|}
<0.10.
$$

AdamW-parallel failure：

$$
\cos(\delta,\Delta\theta_{\text{AdamW}})>0.90.
$$

### 可视化

```text
p1_controllability_heatmap.svg
p1_target_fit_R2_by_subspace.svg
p1_output_displacement_ratio.svg
p1_projection_neutralization_bar.svg
p1_cosine_with_adamw_task.svg
p1_target_residual_spectrum.svg
```

---

## P2：Direct logit oracle target audit

### 目标

判断 output target 本身是否有意义。P2 不更新参数，只在 logits 上施加小 $\Delta z_{\text{target}}$，检查机制指标是否改善。

### 设置

```text
targets = O1-O6
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
eps_z = 0.01,0.03,0.05 relative to logits norm
```

### 必须记录

```text
target_id
dataset
seed
eps_z
CEp99_before
CEp99_after
margin_p10_before
margin_p10_after
ECE_before
ECE_after
NLL_before
NLL_after
accuracy_before
accuracy_after
bad_oracle_step
```

### 判断标准

Target useful：

至少一个机制指标改善且 accuracy 不伤：

$$
Acc_{\text{after}}\geq Acc_{\text{before}}-0.005,
$$

and at least one:

$$
CEp99_{\text{after}}<CEp99_{\text{before}},
$$

$$
MarginP10_{\text{after}}>MarginP10_{\text{before}},
$$

$$
ECE_{\text{after}}\leq ECE_{\text{before}}.
$$

若 target oracle 无效，则该 target 不进入 actuator training。

### 可视化

```text
p2_direct_logit_oracle_effect.svg
p2_target_usefulness_matrix.svg
p2_oracle_accuracy_safety.svg
```

---

## P3：PureKAN actuator implementation and contract audit

### 目标

实现 A1-A8 actuator candidates，并确认它们仍是 strict FC-PureKAN，不是 MLP hidden path 或 external residual。

### 必须记录

```text
candidate_id
actuator_type
basis_formula
edge_owned_params
non_edge_params
ordinary_mlp_path_used
external_residual_used
manual_forward
manual_backward
manual_update
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
basis_condition_number
basis_usage_entropy
dominant_basis_fraction
```

### 判断标准

Contract pass：

```text
edge_owned_params = all trainable params
non_edge_params = 0
ordinary_mlp_path_used = 0
external_residual_used = 0
manual_forward/backward/update = 1
```

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

Interaction retention：

$$
R^2_{\text{pairwise}}\geq0.95.
$$

### 可视化

```text
p3_actuator_contract_heatmap.svg
p3_gradcheck_lollipop.svg
p3_basis_conditioning_bar.svg
p3_pairwise_r2_by_actuator.svg
```

---

## P4：Actuator P4 / P5 base qualification

### 目标

确认 actuator candidate 不破坏 FC-PureKAN base 的 system gate 和 AdamW trainability。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
baseline = same-parameter MLP-match
functional_update = off
```

### 必须记录

P4：

```text
forward_ratio
backward_ratio
step_ratio
compact_memory_ratio
conservative_memory_ratio
kernel_count
actuator_extra_time
```

P5：

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
basis_usage_entropy
actuator_usage_entropy
```

### 判断标准

P4 pass：

$$
forward\leq1.25,
$$

$$
backward\leq1.50,
$$

$$
step\leq1.50,
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

Candidate can enter functional only if both pass.

### 可视化

```text
p4_actuator_system_pareto.svg
p4_actuator_task_gap.svg
p4_actuator_usage_entropy.svg
p4_task_system_joint_dashboard.svg
```

---

## P5：Post-actuator controllability audit

### 目标

复测 P1 中失败的 controllability 指标，判断 actuator 是否真的增加可控性。

### 必须记录

与 P1 相同，并额外记录：

```text
actuator_channel_active_fraction
actuator_channel_snr
actuator_output_sensitivity
actuator_target_fit_gain_vs_LQ
```

### 判断标准

Actuator controllability pass：

$$
R^2_{\text{fit,new}}\geq0.20
$$

or:

$$
R^2_{\text{fit,new}}\geq R^2_{\text{fit,LQ}}+0.10.
$$

and:

$$
r_z\geq0.05.
$$

If no actuator passes, route becomes:

```text
R5-LQFunctionalControllabilityFail
```

### 可视化

```text
p5_controllability_before_after.svg
p5_actuator_target_fit_heatmap.svg
p5_output_displacement_before_after.svg
p5_actuator_sensitivity_spectrum.svg
```

---

## P6：Paired replay causality with actuator

### 目标

只有 P5 controllability pass 后打开。验证 actuator-enabled functional update 是否能击败 controls。

### Branches

```text
AdamW-only
RealFunctional
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledTarget
ShuffledSNRMask
AdamWParallelDirection
```

### Horizons

```text
horizon = 1,5,20,80
```

### 必须记录

```text
event_id
candidate
branch
horizon
dataset
seed
holdout_loss_delta
val_proxy_acc_delta
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
basis_entropy_delta
actuator_usage_delta
step_time
memory_ratio
```

### 判断标准

Paired replay pass：

RealFunctional beats best control on at least one mechanism metric:

$$
CEp99_{\text{real}}\leq CEp99_{\text{best-control}}-0.05|CEp99_{\text{AdamW}}|,
$$

or:

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best-control}}+0.02,
$$

or:

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{best-control}}.
$$

Task safety:

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 可视化

```text
p6_paired_replay_branch_curves.svg
p6_real_vs_controls_mechanism_gain.svg
p6_horizon_effect_heatmap.svg
p6_actuator_event_effects.svg
```

---

## P7：Short-run and full functional re-entry

### 目标

验证 actuator-enabled functional 能否从 local causality 走向多步和 full training advantage。

### P7a short-run 设置

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### P7b full run 设置

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
delta_vs_MLP
delta_vs_AdamW_LQ
delta_vs_QuadraticFeatureMLP
near_pass
full_pass
CEp99
margin_p10
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
actuator_usage_entropy
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
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
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
p7_macro_delta_vs_adamw.svg
p7_kmnist_repair_matrix.svg
p7_seedwise_win_matrix.svg
p7_task_geometry_pareto.svg
p7_ce_tail_margin_panel.svg
p7_ece_nll_panel.svg
p7_actuator_usage_trace.svg
```

---

## P8：Noise / robustness / external-ready gate

### 目标

确认 functional actuator 不是 clean setting 偶然有效。

### Noise 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### External-ready 判断

Functional actuator external-ready if：

```text
functional task-safe = 1
functional mechanism pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
noise robustness pass = 1
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

Strong baseline challenge：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or functional geometry clearly better:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

### 可视化

```text
p8_noise_robustness_curve.svg
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9213.csv
p0_v9212_reproduction.csv
p1_functional_controllability_audit.csv
p2_direct_logit_oracle_target_audit.csv
p3_actuator_contract_gradcheck.csv
p4_actuator_p4_p5_base_qualification.csv
p5_post_actuator_controllability_audit.csv
p6_paired_replay_with_actuator.csv
p7_functional_reentry_with_actuator.csv
p8_robustness_external_ready_gate.csv
functional_event_trace_v9213.csv
actuator_sensitivity_trace_v9213.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_base_gate_regression
F3_v9212_reproduction_unstable
F4_target_oracle_fail
F5_controllability_weak
F6_projection_neutralization
F7_adamw_parallel_direction
F8_actuator_contract_fail
F9_actuator_grad_fail
F10_actuator_p4_fail
F11_actuator_p5_fail
F12_actuator_controllability_no_gain
F13_paired_replay_causality_fail
F14_short_run_or_full_run_no_gain
F15_kmnist_repair_fail
F16_strong_baseline_explains_gain
F17_noise_robustness_fail
F18_fake_or_proxy_violation
F19_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-CurrentLQTargetInvalid:
  direct logit oracle target itself does not improve metrics.

R2-CurrentLQControllabilityWeak:
  target useful but current LQ subspaces cannot fit/move it.

R3-ActuatorBaseQualified:
  actuator candidate passes contract, GradPass, P4 and P5 near-pass.

R4-ActuatorControllabilityPass:
  actuator improves target fit / output displacement enough to reopen paired replay.

R5-ActuatorFunctionalCausalityPass:
  paired replay with actuator beats controls.

R6-ActuatorFunctionalFullAdvantage:
  short-run/full-run shows task-safe macro/KMNIST/geometry gain.

R7-ActuatorHurtsBase:
  actuator improves controllability but destroys P5 near-pass.

R8-NoPureKANActuatorFound:
  no tested actuator preserves P4/P5 and improves controllability.

R9-ReturnToBasisFactory:
  current LQ family lacks functional controllability; next step is basis/primitive redesign.

R10-ExternalReady:
  functional actuator passes full task, robustness, and strong baseline gates.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_actuator_candidate
best_output_target
best_subspace
best_solver
direct_logit_oracle_pass
current_lq_controllability_pass
actuator_contract_pass
actuator_p4_pass
actuator_p5_near_pass
post_actuator_controllability_pass
paired_replay_pass
full_functional_pass
noise_robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9213_controllability_audit
success_v9213_actuator_base
success_v9213_actuator_controllability
success_v9213_functional_advantage
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.12 terminal boundary。

Step 2:
  P1 做 current LQ functional controllability audit。
  先判断 target fit / output displacement / projection neutralization。

Step 3:
  P2 做 direct logit oracle target audit。
  如果 target 自身无效，停止 output-direction路线，重写 target。

Step 4:
  P3 实现 PureKAN actuator candidates A1-A8，并做 contract / gradcheck。

Step 5:
  P4 对 actuator 做 P4 / P5 base qualification。
  任何破坏 near-pass 的 actuator 不进入 functional。

Step 6:
  P5 复测 actuator controllability。
  只有 target fit 或 output displacement 明显提升，才进入 paired replay。

Step 7:
  P6 paired replay with actuator。
  只有 real functional 击败 controls，才进入 short-run/full-run。

Step 8:
  P7 functional re-entry with actuator。

Step 9:
  P8 robustness / external-ready gate。
```

---

## 9. 停止条件

### Minimum success

```text
direct logit target useful
current controllability audited
at least one actuator passes contract + GradPass + P4 + P5
post-actuator controllability improves
no fake/proxy
```

### Functional causality success

```text
Minimum success
+
paired replay beats controls
```

### Functional advantage success

```text
Functional causality success
+
short-run/full-run task-safe mechanism gain
```

### External-ready success

```text
Functional advantage success
+
noise robustness
+
strong baseline challenge
```

### Failure stop

```text
1. v9.2.12 result cannot be reproduced；
2. direct logit targets are not useful；
3. current LQ controllability is weak and no actuator improves it；
4. actuator improves controllability but breaks P4/P5；
5. actuator functional remains control-equivalent；
6. full run gives no macro/KMNIST/geometry/tail improvement；
7. gains are explained by QuadraticFeatureMLP；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：target useful, LQ uncontrollable, actuator works

可以声明：

```text
Functional update required a dedicated PureKAN actuator; original LQ was task-useful but functionally under-controllable.
```

### Case B：target useful, LQ uncontrollable, no actuator works

必须声明：

```text
Current LQ family lacks functional controllability under tested PureKAN actuator designs; return to basis/primitive design.
```

### Case C：target itself invalid

必须声明：

```text
Output-space functional target was poorly specified; redesign target before changing model.
```

### Case D：actuator improves controllability but hurts task

必须声明：

```text
Actuator creates controllability but destroys base trainability; not a viable PureKAN functional route.
```

### Case E：paired replay passes but full run fails

必须声明：

```text
Functional causality is local but not robust across training horizon/seeds.
```

---

## 11. 最终建议

v9.2.13 的一句话策略是：

$$
\boxed{
\text{先问 LQ 是否可控；不可控就设计 PureKAN actuator，而不是继续调旧 functional gate。}
}
$$

当前最关键的问题不是系统实现、AdamW、SNR overhead，也不是 event coverage，而是：

```text
1. 目标 output correction 本身是否有效？
2. 当前 LQ 子空间能不能实现这些 correction？
3. task-safe projection 是不是把所有 functional direction 都中和？
4. 是否需要专门的 edge-owned functional actuator channel？
5. actuator 能否不破坏 P4/P5 near-pass？
6. actuator 后 functional 能否在 paired replay 和 full-run 中超过 controls？
```
