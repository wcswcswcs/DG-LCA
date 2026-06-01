# DG-KAN v9.2.14 P4-Qualified Functional Actuator Closure 与 Absorbable Actuator 完整实验计划

> 本计划基于 v9.2.13 `Functional Controllability 与 PureKAN Actuator Redesign` 的真实 terminal route 制定。  
> v9.2.13 的核心结论非常清楚：**output target 有效，actuator 子空间有强 controllability signal，但没有任何 tested strict FC-PureKAN actuator 同时通过 P4 system gate。**  
> 因此 v9.2.14 不继续调 SNR gate、不继续调 event threshold、不直接打开 paired replay/full functional，也不把 actuator 的强 LS controllability 当成成功。  
> v9.2.14 的核心任务是：
>
> $$
> \boxed{
> \text{把有可控性的 actuator 做到 P4-qualified，或者把 actuator 设计成可吸收到现有 LQ-T2 路径中的低成本 functional actuator。}
> }
> $$
>
> 这里的 actuator 仍必须是 strict FC-PureKAN edge-owned 参数子空间；不允许外部 residual、普通 MLP hidden path、Conv/Former、teacher、distillation、loss modification、label smoothing、sampler/class weight、CPU offload 或 fake/proxy rows。

---

## 0. 当前结果与是否达到目标

### 0.1 v9.2.13 terminal route

v9.2.13 的最终 route 是：

```text
route = R8-NoPureKANActuatorFound
base_candidate = LQ-t2-h256
success_v9213_controllability_audit = true
success_v9213_actuator_base = false
success_v9213_actuator_controllability = false
success_v9213_functional_advantage = false
```

这意味着本轮没有达到 functional advantage，也没有打开 full functional re-entry 或 external fair。  
更准确地说，v9.2.13 证明了三个正向事实和一个硬 blocker。

正向事实：

```text
1. direct logit oracle target 有效；
2. current LQ raw LS 可以拟合 output target；
3. edge-owned actuator subspace 能形成很强 controllability signal。
```

硬 blocker：

```text
没有任何 actuator 同时通过 P4 system gate。
```

所以当前不能声明：

```text
functional update 成功；
actuator route 成功；
PureKAN 已显著 Beyond-MLP；
可以进入 paired replay / full functional / robustness / external fair。
```

### 0.2 v9.2.13 的关键数据

v9.2.13 复现了 v9.2.12 的边界：

```text
source route = R5-ControlEquivalentAgain
source rows = 116640
source pass count = 0
max output displacement ratio = 0.0360087525
max output target fit R2 = 0.0593199465
event coverage = 0.033333 - 0.100000
```

这说明 v9.2.12 的问题不是偶然，也不是 event coverage 没修好。

P1 current LQ controllability audit 显示：

| scope | rows | max R2 | max rz | promising rows |
|---|---:|---:|---:|---:|
| raw LS | `195` | `0.999995` | `49.254015` | `117` |
| task-safe / constrained | `390` | `0.189644` | `0.040227` | `0` |

其中 best safe row 是：

| target | subspace | solver | R2 | rz | projection ratio | holdout delta |
|---|---|---|---:|---:|---:|---:|
| O4 | S6 | SOL2 | `0.189644` | `0.040227` | `0.001601` | `-0.000082` |

这个结果说明：raw output coefficient subspace 能拟合 target，但 task-safe / constrained 后可用位移被压低到：

$$
r_z=0.040227<0.05.
$$

因此 current LQ 的问题不是完全没有可控性，而是 **安全可控性不足**。

P4 actuator base qualification 的关键结果是：

| candidate | forward | backward | step | compact memory | P4 |
|---|---:|---:|---:|---:|---:|
| A0 current | `1.002502` | `0.943293` | `2.248581` | `0.969501` | 0 |
| A1 duplicate T2 | `1.125948` | `1.587883` | `1.222466` | `0.969731` | 0 |
| A2 normalized T2 | `1.327233` | `1.511866` | `1.254456` | `0.969731` | 0 |
| A3 centered T2 | `1.240095` | `1.536814` | `1.242644` | `0.969731` | 0 |
| A4 bounded rational | `1.089077` | `1.514382` | `1.214271` | `0.969731` | 0 |
| A5 piecewise linear 2 | `1.301135` | `1.515704` | `1.253538` | `0.969923` | 0 |
| A6 local RBF | `1.114660` | `1.648638` | `1.222728` | `0.969731` | 0 |
| A7 basis entropy | `1.165031` | `1.544527` | `1.220802` | `0.969731` | 0 |

这个表很重要。它说明：

```text
1. memory 不是 blocker；
2. step 大多不是 blocker；
3. A4/A5/A7 已经非常接近 P4，但 backward 仍未过 <=1.50；
4. A2 forward 和 backward 都略超；
5. A0 current 的 step path 有异常，说明 current/actuator P4 的 phase accounting 也要拆清。
```

P5 post-actuator controllability audit 显示：

```text
rows = 360
actuator_controllability_pass = 273
pass candidates = A1,A2,A3,A4,A5,A6,A7
```

代表性 A4 rows：

| candidate | target | dataset | seed | R2 | rz | gain vs LQ |
|---|---|---|---:|---:|---:|---:|
| A4 bounded rational | O6 | KMNIST | 0 | `0.9999996` | `1.155372` | `+0.810356` |
| A4 bounded rational | O2 | KMNIST | 0 | `0.9999996` | `0.770263` | `+0.810356` |
| A4 bounded rational | O1 | KMNIST | 1 | `0.9999997` | `0.353909` | `+0.810356` |

所以 actuator route 不是没有科学信号。相反，A4 等 actuator 能强拟合 output targets；只是它们没有 P4-qualified base，不能进入 paired replay。

### 0.3 本轮独立判断

v9.2.13 不是 target failure，也不是 pure functional failure。  
它证明：

$$
\boxed{
\text{functional target 有效，actuator 子空间有用；当前失败是 controllability 与 system envelope 没有同时闭合。}
}
$$

更具体：

```text
旧 LQ:
  P4 / P5 near-pass 好，但 safe output displacement 太弱。

actuator:
  output controllability 很强，但 P4 system gate 不过。

下一步:
  不是继续调 functional gate，而是做 actuator system closure / absorbable actuator。
```

---

## 1. 当前处在总计划的什么位置

总计划的终极目标是：

$$
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{External Fair Advantage}
+
\text{Scalable Extension}.
$$

截至 v9.2.13：

| 模块 | 当前状态 |
|---|---|
| Clean FC-PureKAN base | LQ-t2-h256 已通过 strict equivalence audit |
| Graph-free manual training | LQ path manual forward/backward/update 成立 |
| Kernel-native efficiency | LQ base P4 已通过；actuator P4 未通过 |
| AdamW-only task | LQ base robust near-pass，未 full-pass |
| Functional update | target 有效、controllability 有信号，但 actuator 未 P4-qualified |
| External fair | 未打开 |
| PureKANConv / PureKANFormer | 继续 deferred |

因此当前不是 early system stage，也不是 functional success stage。  
最准确定位是：

$$
\boxed{
\text{FC-PureKAN base 已可用；functional update 卡在 P4-qualified controllable actuator。}
}
$$

---

## 2. 当前问题的本质

### 2.1 不是 optimizer 问题

AdamW-only LQ base 已经 robust near-pass。  
如果问题是普通 optimizer 训练不了，我们不会看到 LQ base P4 pass + P5 near-pass。当前功能性问题出现在：

```text
target -> parameter update -> output movement -> paired replay
```

这不是 AdamW lr / weight decay / scheduler 能解决的。

### 2.2 不是 output target 完全错误

Direct logit oracle 已通过，O1/O2/O3/O4/O6 每个 target 的 `27/27` rows 均有 useful signal。  
这说明 “想在 output-space 上改变什么” 不是完全错的。

### 2.3 不是 event coverage 问题

v9.2.12 已经把 sparse event coverage 修到 `0.033333-0.100000`，v9.2.13 继承了这个边界。  
所以不是 v9.2.11 那种 event 全开或全关。

### 2.4 不是 actuator 完全无效

Post-actuator controllability `273/360` pass，A4 bounded rational 在 KMNIST hard target 上 R2 接近 1，rz 可到 1.155。  
所以 actuator 的函数空间作用很强。

### 2.5 真正 blocker

真正 blocker 是：

$$
\boxed{
\text{actuator 的可控性与 P4 system gate 没有同时成立。}
}
$$

也就是：

```text
LQ base:
  系统好，可控弱。

A4/A5/A7:
  可控强，系统略不过。

A6:
  可控可能强，但 backward 明显不过。

当前缺的是：
  P4-qualified actuator 或 absorbable actuator。
```

---

## 3. v9.2.14 整体目标

v9.2.14 的整体目标是：

$$
\boxed{
\text{获得一个 P4-qualified functional actuator，或证明需要从 actuator basis factory 重新设计。}
}
$$

它分两条主线。

### 3.1 主线 A：P4-close near-pass actuator

优先处理 A4/A5/A7 这类 **已经有 controllability 且 P4 只差一点** 的 actuator。

目标是让至少一个 actuator 同时满足：

$$
T_{\text{forward}}\leq1.25T_{\text{MLP-match}},
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP-match}},
$$

$$
T_{\text{step}}\leq1.50T_{\text{MLP-match}},
$$

$$
M_{\text{peak}}\leq1.05M_{\text{MLP-match}},
$$

且保持：

$$
R^2_{\text{fit}}\geq0.20
$$

或：

$$
R^2_{\text{fit}}\geq R^2_{\text{fit,LQ}}+0.10,
$$

并且：

$$
r_z\geq0.05.
$$

### 3.2 主线 B：Absorbable actuator

如果 per-step actuator basis backward 总是把 P4 推出 gate，就设计一种 **可吸收 functional actuator**：

```text
1. functional event 时在 actuator subspace 中求解 output correction；
2. 将该 correction 投影/吸收到现有 LQ-T2 coefficients；
3. 训练主模型仍保持低成本 LQ path；
4. actuator 不作为每步额外 forward/backward 常驻开销。
```

这条线要回答：

$$
\boxed{
\text{能否用 actuator 作为低频 functional solver，而不是长期增加模型成本？}
}
$$

Absorbable actuator 必须满足：

```text
no teacher
no loss modification
no external residual
no non-edge trainable params
no Conv/Former
no CPU offload
no fake/proxy
```

它可以是训练过程中的合法 update rule，但最终参数必须仍能写成 FC-PureKAN edge basis composition。

---

## 4. 核心假设

### H1：A4 bounded rational 是当前最有价值的 actuator candidate

A4 的 P4 指标是：

```text
forward = 1.089077 pass
backward = 1.514382 fail by ~0.014
step = 1.214271 pass
memory = 0.969731 pass
```

同时 A4 在 post-actuator controllability 中有非常强的 KMNIST hard target fit：

```text
R2 ≈ 1.0
rz up to 1.155372
```

H1 成立标准：

经过 backward / derivative / phase closure 后，A4 variant 满足：

$$
backward\leq1.50,
$$

并保持：

$$
R^2_{\text{fit}}\geq0.20,\quad r_z\geq0.05.
$$

### H2：A4/A5/A7 的 P4 failure 主要来自 backward derivative / coefficient gradient，而不是 memory

所有 actuator compact memory 都在 gate 内。A4/A5/A7 的 step 也都在 gate 内，主要超线在 backward。  
因此应优先做 backward attribution，而不是继续改 memory。

H2 成立标准：

P1 phase attribution 显示：

```text
actuator derivative + actuator coefficient grad
占 backward excess 的 >= 60%
```

若不是，则重新定位到 forward/optimizer/update phase。

### H3：Actuator P4 closure 不是小修，而是资格门

因为 P4 不过，P5/P6/P7/P8 都不能打开。  
所以即使 A4 只差 `0.014`，也必须真实通过 gate，不能放宽阈值。

H3 成立标准：

P4 pass 必须来自真实 measured rows，不允许：

```text
derived ratio
proxy row
manual override
threshold relaxation
```

### H4：Absorbable actuator 可能比常驻 actuator 更符合 functional update 本质

Functional update 是低频事件，不一定需要每一步都为 actuator 付 forward/backward 成本。  
如果 actuator 只在 functional event 中求解 correction，然后吸收到现有 LQ-T2 参数，它可能同时保留：

```text
LQ base P4
actuator controllability
functional output effect
```

H4 成立标准：

Absorbable actuator 满足：

$$
R^2_{\text{absorb-fit}}\geq0.20,
$$

$$
r_z\geq0.05,
$$

且 full-step P4 仍按 LQ path 通过。

### H5：如果 A4/A5/A7 P4 close 后仍不能 paired replay beat controls，则 functional target/causality 才是下一个 blocker

只有 P4-qualified controllable actuator 出现后，才能重新判断 functional causality。  
如果 paired replay 仍 control-equivalent，那说明问题转向 output target / causal direction，而不是 actuator system。

---

## 5. Candidate 设计

### 5.1 Baselines

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current FC-PureKAN near-pass base。

LQ1-LQ-t2-h256-fanin-output-scale:
  previous best repair reference。
```

### 5.2 P4-close actuator candidates

```text
A4b-BoundedRational-BranchlessDerivative:
  A4 的 branchless derivative / fused numerator-denominator path。

A4c-BoundedRational-FixedBeta:
  固定 beta，不训练 denominator shape，减少 backward grad。

A4d-BoundedRational-ValueOnlyActuator:
  functional actuator 只更新 output-edge coefficient，不更新 basis shape。

A4e-BoundedRational-FusedCoeffGrad:
  actuator coeffgrad 与 main output coeffgrad 合并。

A5b-PiecewiseLinear2-Branchless:
  piecewise linear 改成 branchless clamp/hat basis。

A5c-PiecewiseLinear2-FixedKnots:
  knots 全固定，只训练 edge coefficients。

A7b-BasisEntropy-LowRank:
  basis entropy actuator 改成低秩 output-edge coefficient。

A7c-BasisEntropy-ValueOnly:
  不训练 entropy basis shape，仅训练 value channel。
```

### 5.3 Absorbable actuator candidates

```text
AB0-ActuatorSolveThenAbsorb-T2:
  用 actuator solve output target，再吸收到 T2 coefficients。

AB1-ActuatorSolveThenAbsorb-Lift:
  吸收到 lift coefficients。

AB2-ActuatorSolveThenAbsorb-LiftPlusT2:
  吸收到 lift + T2 joint subspace。

AB3-ActuatorShadowLowRank:
  低秩 shadow actuator 只在 functional event 中参与 solve，event 后 fold into LQ output coefficients。

AB4-EventOnlyActuatorNoPersistentForward:
  actuator 不参与 normal training forward；只作为 functional update operator。
```

### 5.4 System variants

```text
S0-current:
  current v9.2.13 implementation。

S1-compiled-actuator-forward-backward:
  torch.compile actor path。

S2-fused-derivative-and-coeffgrad:
  derivative 与 coeffgrad fusion。

S3-no-basis-shape-grad:
  fixed basis shape, value-only actuator。

S4-shared-workspace:
  actuator 与 LQ output path 共享 temp buffer。

S5-q90-timing-repeat:
  robust timing q50/q90 protocol。
```

### 5.5 Controls

```text
C0-AdamWOnly
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-ShuffledTarget
C4-AdamWParallelDirection
C5-ActuatorNoAbsorbControl
C6-AbsorbRandomTarget
```

---

## 6. 实验阶段

## P0：复现 v9.2.13 boundary 与 artifact audit

### 目标

确认 v9.2.13 的核心边界稳定存在，避免在测量噪声上继续。

### 必须记录

```text
source_route
direct_logit_oracle_pass
current_lq_raw_max_R2
current_lq_raw_max_rz
current_lq_safe_max_R2
current_lq_safe_max_rz
actuator_contract_pass
actuator_p4_pass_count
post_actuator_controllability_pass_count
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R8-NoPureKANActuatorFound
direct_logit_oracle_pass = 1
actuator_p4_pass_count = 0
post_actuator_controllability_pass_count > 0
fake/proxy = 0
```

### 可视化

```text
p0_v9213_boundary_dashboard.svg
p0_current_vs_actuator_controllability.svg
p0_p4_fail_summary.svg
```

---

## P1：A4/A5/A7 P4 failure phase attribution

### 目标

确定 actuator P4 failure 具体来自哪里。P1 不做新功能，只拆 phase。

### 必须记录

```text
candidate
forward_basis_eval_ms
forward_projection_ms
forward_total_ms
backward_basis_derivative_ms
backward_coeffgrad_ms
backward_output_grad_ms
backward_total_ms
optimizer_update_ms
step_total_ms
compact_memory_ratio
temp_MB
kernel_count
small_kernel_count
unknown_time_fraction
```

### 判断标准

Attribution pass：

$$
unknown\_time\_fraction\leq0.10.
$$

Backward-dominant if：

```text
backward_basis_derivative_ms + backward_coeffgrad_ms
占 backward excess >= 60%
```

Forward-dominant if：

```text
forward_basis_eval_ms + forward_projection_ms
占 forward excess >= 60%
```

### 可视化

```text
p1_p4_phase_waterfall.svg
p1_backward_excess_by_component.svg
p1_forward_excess_by_component.svg
p1_kernel_count_by_actuator.svg
```

---

## P2：A4/A5/A7 P4 closure candidate factory

### 目标

让至少一个 high-controllability actuator 通过 P4。优先 A4，因为它最接近 P4 且 controllability 最强。

### 必跑 candidates

```text
A4b,A4c,A4d,A4e
A5b,A5c
A7b,A7c
```

### 必须记录

```text
candidate
basis_formula
fixed_shape_params
trainable_actuator_params
manual_forward
manual_backward
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
forward_ratio_q50
forward_ratio_q90
backward_ratio_q50
backward_ratio_q90
step_ratio_q50
step_ratio_q90
compact_memory_ratio
conservative_memory_ratio
target_fit_R2
output_displacement_ratio_rz
actuator_usage_entropy
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

Controllability retention：

$$
R^2_{\text{fit}}\geq0.20
$$

or:

$$
R^2_{\text{fit}}\geq R^2_{\text{fit,LQ}}+0.10,
$$

and:

$$
r_z\geq0.05.
$$

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

### 可视化

```text
p2_actuator_p4_pareto.svg
p2_controllability_vs_backward.svg
p2_a4_family_before_after.svg
p2_gradcheck_by_candidate.svg
```

---

## P3：Absorbable actuator audit

### 目标

判断是否可以把 actuator 的 controllable output correction 吸收到低成本 LQ path，避免常驻 actuator P4 overhead。

### 方法

对每个 event target 先在 actuator shadow subspace 中求解：

$$
\delta_a^\star
=
\arg\min_{\delta_a}
\|J_a\delta_a-\Delta z_{\text{target}}\|^2+\lambda\|\delta_a\|^2.
$$

再求一个 LQ absorption update：

$$
\delta_{\text{LQ}}^\star
=
\arg\min_{\delta_{\text{LQ}}}
\|J_{\text{LQ}}\delta_{\text{LQ}}-J_a\delta_a^\star\|^2+\lambda\|\delta_{\text{LQ}}\|^2.
$$

记录 absorption fit：

$$
R^2_{\text{absorb}}
=
1-
\frac{\|J_{\text{LQ}}\delta_{\text{LQ}}^\star-J_a\delta_a^\star\|^2}
{\|J_a\delta_a^\star\|^2+\epsilon}.
$$

### 必须记录

```text
candidate
target
dataset
seed
shadow_actuator_R2
shadow_actuator_rz
absorb_subspace
absorb_R2
absorb_rz
holdout_delta
bad_event
step_ratio_q90
memory_ratio
functional_event_overhead
```

### 判断标准

Absorbable pass：

$$
R^2_{\text{absorb}}\geq0.20,
$$

$$
r_z\geq0.05,
$$

$$
BadEventRate\leq0.05,
$$

and normal LQ P4 remains pass.

If absorption improves over current LQ safe best：

$$
R^2_{\text{absorb}}\geq0.189644+0.10
$$

or:

$$
r_z\geq0.05.
$$

### 可视化

```text
p3_absorb_fit_heatmap.svg
p3_shadow_vs_absorbed_displacement.svg
p3_absorb_bad_event_rate.svg
p3_absorb_overhead.svg
```

---

## P4：P4-qualified actuator base qualification

### 目标

只有 P2 或 P3 产生 survivor 后打开。验证 survivor 不破坏 AdamW-only base。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
baseline = same-parameter MLP-match
```

### 必须记录

```text
candidate
dataset
seed
KAN_acc
MLP_match_acc
delta_vs_mlp
near_pass
train_loss
test_loss
CEp99
margin_p10
ECE
NLL
basis_usage_entropy
actuator_usage_entropy
forward_ratio_q90
backward_ratio_q90
step_ratio_q90
compact_memory_ratio
```

### 判断标准

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

P4 remains pass：

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

### 可视化

```text
p4_actuator_base_task_gap.svg
p4_actuator_p4_task_joint.svg
p4_actuator_usage_entropy.svg
```

---

## P5：Post-closure paired replay causality

### 目标

重新打开 paired replay，但只对 P4/P5-qualified actuator survivor。验证 real functional 是否超过 controls。

### Branches

```text
AdamW-only
RealFunctional
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledTarget
AdamWParallelDirection
ActuatorNoAbsorbControl
AbsorbRandomTarget
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
p5_paired_replay_branch_curves.svg
p5_real_vs_controls_mechanism_gain.svg
p5_horizon_effect_heatmap.svg
p5_actuator_event_effects.svg
```

---

## P6：Short-run and full functional re-entry

### 目标

验证 actuator-enabled functional 是否能从 local causality 走向多步和 full training advantage。

### P6a short-run

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### P6b full run

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
p6_macro_delta_vs_adamw.svg
p6_kmnist_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_ece_nll_panel.svg
p6_actuator_usage_trace.svg
```

---

## P7：Noise / robustness / external-ready gate

### 目标

确认 functional actuator 不是 clean setting 偶然有效。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
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

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9214.csv
p0_v9213_reproduction.csv
p1_actuator_p4_failure_attribution.csv
p2_p4_closure_candidate_factory.csv
p3_absorbable_actuator_audit.csv
p4_p4_qualified_actuator_base_qualification.csv
p5_paired_replay_after_actuator_closure.csv
p6_functional_reentry_after_actuator_closure.csv
p7_noise_robustness_external_ready.csv
functional_event_trace_v9214.csv
actuator_phase_trace_v9214.csv
absorbable_actuator_trace_v9214.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9213_reproduction_unstable
F3_p4_failure_attribution_incomplete
F4_actuator_grad_fail
F5_actuator_p4_fail
F6_actuator_controllability_lost
F7_absorbable_actuator_fail
F8_actuator_base_trainability_fail
F9_paired_replay_causality_fail
F10_full_run_no_gain
F11_kmnist_repair_fail
F12_strong_baseline_explains_gain
F13_noise_robustness_fail
F14_system_overhead_fail
F15_fake_or_proxy_violation
F16_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-A4P4Closure:
  A4 bounded rational family closes P4 and retains controllability.

R2-A5OrA7P4Closure:
  A5/A7 family closes P4 and retains controllability.

R3-AbsorbableActuatorPass:
  shadow actuator + absorption keeps LQ P4 and improves controllability.

R4-ActuatorP4ClosedBaseQualified:
  P4-qualified actuator also keeps AdamW near-pass.

R5-ActuatorFunctionalCausalityPass:
  paired replay real functional beats controls.

R6-ActuatorFunctionalFullAdvantage:
  short/full run improves macro/KMNIST/geometry/tail without task/system harm.

R7-ControllabilityLostDuringP4Closure:
  actuator becomes P4-pass only by losing target fit / rz.

R8-P4CloseButBaseTrainabilityFail:
  actuator passes system but destroys P5 near-pass.

R9-NoP4QualifiedActuator:
  no tested actuator closes P4.

R10-AbsorptionOnlyButNoCausality:
  absorbable actuator improves fit but paired replay remains control-equivalent.

R11-ReturnToBasisFactory:
  neither P4-close nor absorbable actuator works; need new basis/primitive.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_actuator_candidate
best_actuator_family
best_absorbable_candidate
best_output_target
best_solver
p4_closure_pass
controllability_retained
absorbable_pass
actuator_base_near_pass
paired_replay_pass
full_functional_pass
noise_robustness_pass
strong_baseline_pass
external_ready
forward_ratio_q90
backward_ratio_q90
step_ratio_q90
memory_ratio
target_fit_R2
output_displacement_ratio_rz
primary_blocker
next_required_implementation
success_v9214_p4_qualified_actuator
success_v9214_absorbable_actuator
success_v9214_functional_advantage
success_v9214_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.13 boundary。

Step 2:
  P1 拆 A4/A5/A7 P4 failure phase。
  如果 unknown_time_fraction > 0.10，先修 profiler。

Step 3:
  P2 优先做 A4 family P4 closure。
  A4 是当前最有 controllability 且最接近 P4 的 actuator。

Step 4:
  并行做 P3 absorbable actuator。
  这不是绕过 P4，而是把 actuator 成本变成 event-time update operator。

Step 5:
  若 P2 或 P3 产生 survivor，执行 P4 base qualification。
  没有 P4/P5-qualified survivor，不打开 paired replay。

Step 6:
  P5 paired replay。
  必须超过 controls。

Step 7:
  P6 short/full functional re-entry。

Step 8:
  P7 robustness / external-ready gate。
```

---

## 10. 停止条件

### Minimum success

```text
v9.2.13 boundary reproduced
P4 failure attributed
at least one P4-qualified controllable actuator or absorbable actuator
no fake/proxy
```

### Functional causality success

```text
Minimum success
+
paired replay real functional beats controls
```

### Functional advantage success

```text
Functional causality success
+
short/full run task-safe macro/KMNIST/geometry/tail gain
```

### Failure stop

```text
1. v9.2.13 boundary cannot be reproduced；
2. P4 failure attribution incomplete；
3. all A4/A5/A7 closure candidates fail P4；
4. P4 closure destroys controllability；
5. absorbable actuator cannot improve safe rz / fit；
6. P4-qualified actuator destroys AdamW near-pass；
7. paired replay remains control-equivalent；
8. full run gives no macro/KMNIST/geometry/tail gain；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：A4 family closes P4 and retains controllability

可以声明：

```text
P4-qualified bounded rational actuator found; functional paired replay can reopen.
```

但不能直接声明 functional advantage，除非 P5/P6 通过。

### Case B：P4 closure works only by weakening actuator

必须声明：

```text
System closure destroyed controllability; actuator design is not useful.
```

### Case C：Absorbable actuator works

可以声明：

```text
Functional actuator need not be a persistent per-step basis channel; it can be an event-time PureKAN update operator absorbed into LQ.
```

### Case D：All actuator paths fail

必须声明：

```text
Current LQ/T2 family lacks a P4-qualified controllable actuator; next step should return to basis/primitive factory, not functional gate tuning.
```

### Case E：paired replay fails after P4 closure

必须声明：

```text
Actuator controllability exists, but output target / causal direction still does not translate into functional advantage.
```

---

## 12. 最终建议

v9.2.14 的一句话策略是：

$$
\boxed{
\text{优先把 A4/A5/A7 的强 controllability 做成 P4-qualified actuator；若常驻 actuator 太贵，就转向 absorbable actuator。}
}
$$

当前最关键的不是继续调 SNR、event、optimizer，也不是提前研究 Conv/Former，而是：

```text
1. A4/A5/A7 为什么只差 backward gate？
2. 能否通过 fixed-shape / value-only / fused coeffgrad 关闭 P4？
3. 关闭 P4 后 controllability 是否保留？
4. actuator 是否必须常驻在模型中，还是可以作为 event-time solver 被吸收到 LQ？
5. 一旦获得 P4-qualified controllable actuator，paired replay 是否终于能超过 controls？
```
