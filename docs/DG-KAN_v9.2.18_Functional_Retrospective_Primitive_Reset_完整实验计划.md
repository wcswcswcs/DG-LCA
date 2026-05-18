# DG-KAN v9.2.18 Functional Update 复盘桥接与 Primitive Reset 决策完整实验计划

> 本计划基于 v8.3-v8.7 的 functional update 成功经验，以及 v9.2.9-v9.2.17 的严格 FC-PureKAN functional 失败链制定。  
> 本轮核心问题不是“functional update 是否永远失败”，而是要回答：
>
> $$
> \boxed{
> \text{v8.x 的 functional 成功到底来自 functional update 本身，还是来自 transitional architecture / LR-like extra step / timing protocol？}
> }
> $$
>
> 以及：
>
> $$
> \boxed{
> \text{如果 v9.x 当前 LQ/A4/A7c family 提取不出 functional advantage，下一步应如何重建 primitive / basis / actuator？}
> }
> $$
>
> 本计划继续遵守：no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal / margin / calibration loss、no sampler / class weight、no CPU offload、no fake / proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv / PureKANFormer 继续 deferred。

---

## 0. 当前结论：functional update 真的失败了吗？

### 0.1 一句话结论

Functional update **没有作为研究方向被彻底证伪**。  
但在当前 v9.x strict FC-PureKAN LQ/A4/A7c family 中，已经有两个很强的负结论：

$$
\boxed{
\text{hand-designed output target + actuator + paired replay 路线失败。}
}
$$

以及：

$$
\boxed{
\text{signal-aligned metric 没有超过 AdamWParallel / scalar LR controls。}
}
$$

所以当前不能说：

```text
functional update 成功；
functional update 是当前 strict FC-PureKAN 的核心优势；
可以进入 external fair；
可以扩展 PureKANConv / PureKANFormer。
```

但也不能说：

```text
functional update 永远失败；
v8.x 成功全部无意义；
KAN functional training 方向应直接放弃。
```

更准确的状态是：

$$
\boxed{
\text{v8.x 的 functional 成功需要被 v9-style controls 重新审计；v9.x 当前 family 需要 primitive / basis / actuator reset。}
}
$$

### 0.2 为什么不是“彻底失败”

v8.3-v8.7 的结果确实显示过 functional update 的真实价值：

```text
v8.3:
  system-gated functional update re-entry；
  FT7 role-wise guarded functional update 完成 minimum success。

v8.4:
  high-rep timing-stabilized protocol 下，functional causality / role mechanism 有正信号；
  但 formal timing / original 50/200 style 仍有限制。

v8.5-v8.7:
  transitional route 进入 KANbeFair 外部公平；
  v8.7 formal selected route 在 FMNIST / KMNIST 上通过 external fair，
  并保留 curvature / functional causality 信号。
```

但是这些成功并不是 strict Clean FullEdge PureKAN 成功。v8.x 的强路线结构更接近：

```text
packed / cached linear-SiLU stack
+
poly2_silu KAN-style head
+
manual AdamW-equivalent update
+
adaptive / role-wise functional update
```

它是 **transitional DG-KAN functional success**，不是最终 Clean FullEdge PureKAN success。

### 0.3 当前 v9.x 的真实失败边界

v9.x 严格路线已经逐步排除了多个可能解释：

```text
LQ base:
  strict FC-PureKAN equivalence pass；
  P4 system pass；
  P5 robust near-pass；
  但 full-pass 未达成。

SNR / functional predictor:
  low-cost SNR 可用；
  one-step safe direction 可找到；
  但 short-run / paired replay 没有稳定因果收益。

Output-space functional:
  direct oracle target 有用；
  actuator 可控性强；
  P4-qualified actuator 已找到；
  但 RealFunctional 打不过 controls。

Signal-aligned metric:
  best control 是 AdamWParallel / LR-like control；
  raw survivor 扣除 LR/extra-AdamW equivalence 后 effective survivor = 0。
```

因此当前 v9.x 的真实结论是：

$$
\boxed{
\text{当前 LQ/A4/A7c family 中，没有提取出独立于 AdamW/LR controls 的 functional advantage。}
}
$$

---

## 1. 本轮总体目标

v9.2.18 的总体目标不是继续给旧 functional 加补丁，而是建立一个 **functional update 复盘桥接实验**：

$$
\boxed{
\text{用同一套 v9-style controls 重新审计 v8.x 成功，并决定 functional 是否保留为主线。}
}
$$

本轮要回答六个问题。

### Q1：v8.x functional success 能否经受 v9-style controls？

v8.x 当时已经做过 NoOp / Random / role controls，但没有系统加入后来证明非常关键的：

```text
AdamWParallelDirection
scalar LR controls
extra AdamW trust-ratio controls
ShuffledTarget
control-derived target
control-contrastive solver
```

如果 v8.x 在这些强 controls 下仍然有优势，则说明 functional update 的确有独立信号，只是 v9.x strict primitive 没有承载住。

如果 v8.x 也输给 AdamWParallel / LR controls，则应重新解释 v8.x 成功：它可能主要来自 under-step、timing protocol、architecture residual interface 或 branch activity，而不是独立 functional direction。

### Q2：v8.x 成功和 v9.x 失败的结构差异是什么？

重点比较：

```text
v8.x:
  residual / cached / linear-SiLU stack；
  KAN-style head；
  branch activity 明确；
  role-wise FT7 update；
  functional event 与 stack/head roles 绑定。

v9.x:
  strict FC-PureKAN LQ；
  linear lift + T2 edge basis；
  A4/A7 actuator；
  output-space target / actuator correction；
  branch/residual interface 不存在或不同。
```

需要判断 functional update 是否依赖一个“可安全扰动的 residual branch interface”。如果是，那么 strict FullEdge 的 functional actuator 也必须内生地提供类似功能，而不是只靠 output target fitting。

### Q3：v8.x functional 是 geometry advantage，还是 scalar optimizer effect？

必须把 v8.x 的 FT7 和以下 controls 比较：

```text
AdamW-only
AdamWParallel same norm
LRScale 1.003 / 1.01 / 1.03 / 1.10
NoOp matched overhead
Random matched norm
Shuffled role mask
Inverted role mask
```

如果 FT7 只等价于更大 LR，那么不能说它是 functional advantage。

### Q4：当前 v9.x 是否应该继续 functional，还是回到 primitive/basis？

如果 v8.x 在强 controls 下依然通过，而 v9.x 不通过，则说明：

```text
functional update concept survives；
current v9 primitive lacks functional actuatability；
next step = primitive/basis redesign with functional actuatability gate。
```

如果 v8.x 也不通过，则说明：

```text
current functional evidence需要降级；
next step = AdamW-only full-pass + basis/primitive factory；
functional 暂停为 diagnostic。
```

### Q5：新的 primitive 需要什么资格？

新 primitive 不只要 P4/P5，还要有 functional actuatability：

$$
\boxed{
\text{P4 + P5 near-pass + non-AdamW controllability + control-resistant paired replay。}
}
$$

### Q6：是否可以开始 PureKANConv / PureKANFormer？

本轮仍然不打开。只有当 FC-PureKAN base 或 functional route 明确达到：

```text
P4 pass
P5 near-pass or full-pass
functional advantage pass or AdamW full-pass with strong mechanism
external fair ready
```

才允许解锁 Conv / Former。

---

## 2. 核心假设

### H1：v8.x functional success 可能是真实的，但它依赖 transitional architecture

v8.x 的 functional update 可能不是一般形式的 output correction，而是利用了 transitional architecture 中的 residual-like / role-wise branch：

$$
h_{k+1}=h_k+\alpha_k F_k(h_k).
$$

这种结构给 functional update 提供了天然 actuator：

```text
stack branch；
head branch；
event-triggered branch；
guarded update branch。
```

H1 成立标准：

v8.x FT7 在 v9-style controls 下仍然满足：

$$
Acc_{\text{FT7}}\geq Acc_{\text{AdamW}}-0.005,
$$

并且至少一个机制指标优于 AdamWParallel / LR controls：

$$
CEp99_{\text{FT7}}<CEp99_{\text{AdamWParallel}},
$$

或：

$$
MarginP10_{\text{FT7}}>MarginP10_{\text{AdamWParallel}},
$$

或：

$$
Curvature_{\text{FT7}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

### H2：v8.x success 也可能只是 LR / extra-step effect

v9.2.17 显示 AdamWParallel / scalar LR controls 能解释当前 signal-aligned functional raw survivors。H2 要求把同样的强 controls 加到 v8.x。

H2 成立标准：

如果：

$$
MetricGain_{\text{FT7}}\leq MetricGain_{\text{best LR control}}+\epsilon,
$$

则 v8.x functional success 不能再解释为独立 functional advantage，只能解释为 optimizer-scale / under-step / schedule effect。

默认阈值：

```text
epsilon_CEp99 = 0.02 relative
epsilon_margin = 0.005 absolute
epsilon_ECE = 0.005 absolute
epsilon_curvature = 0.02 relative
```

### H3：v9.x 当前失败来自 primitive 缺少 functional actuatability，而不是 functional 概念错误

当前 LQ/A4/A7c 已有：

```text
P4 system pass；
P5 near-pass；
actuator target fit；
output controllability。
```

但仍没有 control-resistant causality。H3 假设它的问题不是“不能动”，而是“动得没有独立信息”；也就是缺少 non-AdamW functional actuatability。

H3 成立标准：

新 primitive 必须在 paired replay 中满足：

$$
RealFunctional > AdamWParallel
$$

以及：

$$
RealFunctional > best\ LR\ control.
$$

否则不算 functional actuatability pass。

### H4：如果 v8.x 成功经受 controls，下一步应把 v8 的机制迁移到 strict FC-PureKAN

迁移对象不是 linear-SiLU stack 本身，而是机制：

```text
role-wise branch activity；
event-triggered update；
residual-like safe interface；
branch ratio controller；
effective derivative scale controller。
```

H4 成立标准：

在 strict FC-PureKAN 中引入 edge-owned branch/activity controller 后，必须保持：

```text
no external residual；
no ordinary MLP hidden path；
all params edge-owned；
P4 pass；
P5 near-pass；
paired replay control pass。
```

### H5：如果 v8.x 也被 LR controls 解释，应暂停 current functional line

若 v8.x FT7 也无法击败 AdamWParallel / LR controls，则 route 必须写：

```text
R8-FunctionalEvidenceDowngraded_ReturnToPrimitiveBasis
```

而不是继续写 v9.2.19 functional target patch。

---

## 3. 实验阶段

## P0：历史成功与当前失败的统一审计

### 目标

建立 v8.x 与 v9.x 的统一对照表，避免把不同合同下的成功混为一谈。

### 需要审计的历史路线

```text
v8.3:
  FT7 role-wise functional minimum success。

v8.4:
  high-rep timing-stabilized minimum；
  causality / role mechanism positive；
  formal timing not fully closed。

v8.5:
  KANbeFair MNIST external fair functional route。

v8.7:
  selected KW4 hidden28 external fair route on FMNIST/KMNIST。

v9.2.6-v9.2.7:
  LQ strict FC-PureKAN P4 + P5 near-pass。

v9.2.14-v9.2.17:
  P4-qualified actuator；
  no control-resistant functional advantage；
  AdamWParallel / LR equivalence.
```

### 必须记录

```text
version
route
candidate
architecture_family
strict_full_edge_purekan
transitional_route
functional_type
controls_used
adamwparallel_control_used
lr_control_used
p4_pass
p5_nearpass
p5_fullpass
functional_pass
external_fair_pass
broad_strong_claimed
no_fake_proxy
```

### 判断标准

P0 pass 要求所有历史结论重新标注为：

```text
Strict FC-PureKAN success
Transitional functional success
Diagnostic assisted success
System-only success
No functional advantage
```

不得把 transitional success 写成 strict FullEdge PureKAN success。

### 可视化

```text
p0_version_route_lattice.svg
p0_contract_difference_table.md
p0_success_definition_matrix.svg
```

---

## P1：v8.x route 的 v9-style control replay

### 目标

用 v9.2.17 的强 control protocol 重新审计 v8.7 selected route 与 v8.4/v8.3 FT7 route。

### 设置

候选：

```text
V8-FT7-KW6-hidden68
V8-FT7-KW4-hidden28
V8-Adaptive-FT-P-selected
V8-AdamW-only-base
```

Controls：

```text
AdamWOnly
FT7 / AdaptiveFunctional
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledRoleMask
InvertedRoleMask
AdamWParallelSameNorm
AdamWParallelTrustRatio-0.003
AdamWParallelTrustRatio-0.01
AdamWParallelTrustRatio-0.03
LRScale-1.003
LRScale-1.01
LRScale-1.03
LRScale-1.10
```

Datasets：

```text
MNIST
Fashion-MNIST
KMNIST
```

Seeds：

```text
0,1,2,3,4
```

Horizon：

```text
1,5,20,80
```

### 必须记录

```text
candidate
dataset
seed
horizon
branch
test_acc
delta_vs_adamw
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature
local_lipschitz
functional_event_count
event_coverage
step_ratio_q90
memory_ratio
control_rank
real_beats_adamwparallel
real_beats_best_lr
```

### 判断标准

v8 functional survives strong controls if：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

and at least one mechanism metric beats both AdamWParallel and best LR control:

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{best LR}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{best LR}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

If not, v8 functional is reclassified as:

```text
LR-equivalent or optimizer-scale effect
```

### 可视化

```text
p1_v8_functional_vs_strong_controls.svg
p1_control_rank_by_dataset.svg
p1_ft7_vs_adamwparallel_margin_ce.svg
p1_curvature_vs_accuracy_controls.svg
```

---

## P2：v8 functional mechanism attribution

### 目标

如果 v8 functional 仍然有效，要找出它为什么有效；如果无效，要知道它被什么 control 解释掉。

### 必须记录

```text
role
role_update_norm
role_snr
role_curvature
role_event_frequency
branch_ratio
effective_derivative_scale
cos_functional_adamw
cos_functional_random
cos_functional_lrcontrol
functional_norm_vs_adamw
pre_holdout_loss
post_holdout_loss
CEp99_delta
margin_delta
curvature_delta
```

### 机制指标定义

branch ratio：

$$
r_{\text{branch}}
=
\frac{\|\alpha F(h)\|}
{\|h\|+\epsilon}.
$$

effective derivative scale：

$$
s_{\text{eff}}
=
|\alpha|\cdot \phi'_{p95}.
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

### 判断标准

v8 functional mechanism is independent if：

```text
cos with AdamW is not always > 0.95
functional beats LR controls
branch activity is non-trivial
curvature/tail improvement is not reproduced by NoOp/Random/LR
```

v8 functional is LR-equivalent if：

```text
cos with AdamW > 0.95
best LR control matches or beats functional
branch activity does not explain gains
```

### 可视化

```text
p2_rolewise_functional_mechanism.svg
p2_branch_ratio_vs_gain.svg
p2_cosine_with_adamw_histogram.svg
p2_effective_derivative_scale_trace.svg
```

---

## P3：v8-v9 mechanism transfer audit

### 目标

判断 v8 的成功机制能否迁移到 strict FC-PureKAN，而不是直接复制 transitional architecture。

### Transfer candidates

```text
T0-LQ-AdamWOnly:
  v9 LQ base。

T1-LQ-RoleSNRMetric:
  role-aware signal metric。

T2-LQ-BranchActivityController:
  edge-owned branch activity controller，不引入 external residual。

T3-LQ-EffectiveDerivativeController:
  控制 T2 / actuator derivative p95。

T4-A7c-RoleSNRMetric:
  在 P4-qualified A7c actuator 上做 role-aware metric。

T5-A4e-RoleSNRMetric:
  在 bounded rational actuator 上做 role-aware metric。

T6-LQ-FT7StyleEventGuard:
  借鉴 FT7 event guard，但所有参数必须 edge-owned。
```

### 必须记录

```text
candidate
full_edge_equivalence_pass
external_residual_used
ordinary_mlp_path_used
edge_owned_param_fraction
p4_forward_q90
p4_backward_q90
p4_step_q90
p4_memory
p5_nearpass
paired_replay_real_vs_adamwparallel
paired_replay_real_vs_lrcontrol
branch_ratio
effective_derivative_scale
CEp99
margin_p10
ECE
curvature
```

### 判断标准

Mechanism transfer pass if：

$$
P4=1,
$$

$$
P5_{\text{nearpass}}=1,
$$

and:

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

If transfer fails but v8 survives controls, route becomes:

```text
FunctionalConceptValid_PrimitiveActuatorMismatch
```

If transfer fails and v8 fails controls:

```text
FunctionalEvidenceDowngraded_ReturnToPrimitiveBasis
```

### 可视化

```text
p3_v8_to_v9_transfer_matrix.svg
p3_mechanism_transfer_pareto.svg
p3_purekan_contract_heatmap.svg
```

---

## P4：Primitive / basis factory reset

### 目标

如果 current LQ/A4/A7c family 没有 functional advantage，重新设计 FC-PureKAN primitive。新 primitive 不只追 P4/P5，还要追 functional actuatability。

### Candidate families

```text
B0-LQ-T2-current:
  当前 baseline。

B1-LQ-centered-T2:
  centered quadratic basis。

B2-LQ-normalized-T2:
  variance-normalized T2。

B3-LQ-Legendre2-only:
  orthogonal low-order polynomial。

B4-LQ-normalized-Legendre2:
  normalized Legendre2。

B5-LQ-bounded-rational-base:
  bounded rational 作为 base basis，而不只是 actuator。

B6-LQ-piecewise-linear2-base:
  piecewise-linear local basis。

B7-LQ-shared-RBF4-base:
  shared-center RBF local basis。

B8-LQ-mixed-T2-rational:
  T2 + bounded rational edge-basis。

B9-LQ-functional-actuator-channel:
  task basis + functional actuator channel，但必须 P4/P5。
```

### 必须记录

```text
basis_family
basis_formula
derivative_formula
conditioning_pass
basis_condition_number
dominant_basis_fraction
dead_basis_fraction
synthetic_pairwise_R2
local_bump_R2
GradRelErrMax
GradCosMin
p4_forward_q90
p4_backward_q90
p4_step_q90
p4_memory
p5_nearpass_rate
macro_delta
KMNIST_delta
functional_actuatability_R2
non_adamw_output_displacement
paired_replay_vs_adamwparallel
paired_replay_vs_lrcontrol
```

### 基础晋级标准

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

Functional actuatability gate：

$$
R^2_{\text{functional target fit}}\geq0.20,
$$

$$
r_z\geq0.05,
$$

and:

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

### 可视化

```text
p4_basis_factory_pareto.svg
p4_basis_conditioning_heatmap.svg
p4_functional_actuatability_matrix.svg
p4_kmnist_delta_vs_basis.svg
```

---

## P5：AdamW-only full-pass repair without functional

### 目标

防止 functional 掩盖 base 不够强的问题。LQ/A7c near-pass 但未 full-pass，仍需要独立修 base。

### Allowed repairs

```text
orthogonal lift init
fan-in output scale
centered / normalized basis
basis-balanced init
hidden bracket h224/h256/h288
bounded rational base
piecewise local base
```

Forbidden：

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
p4_step_q90
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
p5_adamw_fullpass_gap.svg
p5_kmnist_miss_rows.svg
p5_ce_tail_margin.svg
p5_basis_entropy_vs_gap.svg
```

---

## P6：路线决策

### 目标

根据 P1-P5 明确决定下一步，不再无限 functional patching。

### Route cases

```text
R1-v8FunctionalSurvivesStrongControls:
  v8 functional beats AdamWParallel / LR controls。

R2-v8FunctionalIsLREquivalent:
  v8 functional success explained by scalar LR / extra-step controls。

R3-FunctionalConceptValid_PrimitiveMismatch:
  v8 survives controls, v9 fails transfer；说明 primitive 缺 functional actuatability。

R4-v9SignalFunctionalPass:
  v9 strict FC-PureKAN signal functional beats controls。

R5-AdamWFullPassNoFunctional:
  base primitive reaches full-pass without functional。

R6-ReturnToPrimitiveBasisFactory:
  current functional family exhausted；focus on primitive/basis。

R7-ExternalFairReady:
  strict FC-PureKAN base/function route ready for external validation。

R8-FunctionalPaused:
  no current evidence supports continuing functional as mainline。
```

### route_decision.json 必须记录

```text
route
v8_survives_controls
v8_lr_equivalent
v9_transfer_pass
primitive_factory_pass
adamw_fullpass
functional_actuatability_pass
external_ready
primary_blocker
next_required_implementation
success_v9218_functional_retained
success_v9218_primitive_reset
success_v9218_external_ready
```

---

## 4. Required artifacts

```text
run_manifest.json
contract_audit_v9218.csv
p0_history_unified_audit.csv
p1_v8_strong_control_replay.csv
p2_v8_functional_mechanism_attribution.csv
p3_v8_v9_transfer_audit.csv
p4_primitive_basis_factory_reset.csv
p5_adamw_only_fullpass_repair.csv
p6_route_decision.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v8_reproduction_fail
F3_v8_control_equivalent
F4_v8_mechanism_unattributed
F5_v9_transfer_fail
F6_purekan_contract_fail
F7_p4_system_fail
F8_p5_nearpass_fail
F9_functional_actuatability_fail
F10_adamw_fullpass_fail
F11_external_not_ready
F12_fake_or_proxy_violation
F13_artifact_missing
```

---

## 5. 第一轮执行顺序

```text
Step 1:
  P0 统一审计 v8/v9 成功定义，明确哪些是 transitional，哪些是 strict FC-PureKAN。

Step 2:
  P1 先重测 v8 functional route under v9-style controls。
  这是回答“functional 是否真的失败”的关键。

Step 3:
  P2 做 v8 mechanism attribution。
  如果 v8 也被 LR controls 解释，则 functional 证据降级。

Step 4:
  P3 做 v8 -> v9 mechanism transfer。
  如果 v8 有独立机制，尝试迁移到 strict FC-PureKAN。

Step 5:
  P4 启动 primitive / basis factory reset。
  新 primitive 必须包含 functional actuatability gate。

Step 6:
  P5 并行做 AdamW-only full-pass repair。
  不能用 functional 掩盖 base 未 full-pass。

Step 7:
  P6 路线决策。
  不允许继续无边界地改 functional target。
```

---

## 6. 停止条件

### 成功停止

Functional retained：

```text
v8 survives strong controls
and either v9 transfer passes or new primitive passes functional actuatability
```

Primitive reset success：

```text
new primitive P4 pass
P5 near-pass or full-pass
functional actuatability pass or AdamW full-pass
```

External-ready：

```text
strict FC-PureKAN base/function route
P4 pass
P5 near-pass/full-pass
controls pass
strong baseline pass
robustness pass
```

### 失败停止

```text
1. v8 route cannot reproduce；
2. v8 functional is LR-equivalent；
3. v9 transfer fails；
4. primitive factory has no P4/P5 candidate；
5. functional actuatability fails for all candidates；
6. AdamW-only full-pass repair fails；
7. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 7. 最终解释规则

### Case A：v8 functional 经受强 controls，v9 transfer 失败

应该声明：

```text
Functional update concept remains valid, but current strict FC-PureKAN primitive lacks the actuator/interface needed to realize it.
```

下一步应继续 primitive/basis design，而不是否定 functional。

### Case B：v8 functional 也被 LR controls 解释

应该声明：

```text
Previous functional success must be downgraded; it may reflect optimizer scale / under-step / architecture timing rather than independent functional update.
```

这时 functional 暂停为 diagnostic，主线回到 primitive / AdamW full-pass。

### Case C：v9 transfer 成功

可以声明：

```text
Functional update has been recovered in strict FC-PureKAN under strong controls.
```

但仍需 full 10-seed / robustness / external fair 才能说 Beyond-MLP。

### Case D：新 primitive AdamW full-pass，但 functional 不通过

应该声明：

```text
Primitive design improved the base model; functional advantage remains unproven.
```

### Case E：新 primitive functional actuatability 通过

可以声明：

```text
Functional update becomes viable again through a primitive that exposes non-AdamW controllable function-space directions.
```

---

## 8. 最终建议

v9.2.18 的一句话策略是：

$$
\boxed{
\text{先用 v9-style controls 重新审计 v8.x functional 成功；若它仍成立，就把机制迁移到新的 strict FC-PureKAN primitive；若不成立，就停止当前 functional 主线，回到 primitive/basis 与 AdamW full-pass。}
}
$$

这一步的本质不是再多做一个 functional target，而是明确：

```text
functional update 是真实独立机制；
还是 optimizer scale / architecture-specific effect；
还是需要新的 primitive 才能显现。
```
