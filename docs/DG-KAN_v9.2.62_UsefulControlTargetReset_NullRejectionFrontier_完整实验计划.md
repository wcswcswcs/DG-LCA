# DG-KAN v9.2.62 Useful-Control Target Decomposition 与 Coverage-Constrained Null-Rejection Frontier 完整实验计划

> 本计划基于 v9.2.61 `Value-Risk Orthogonalization 与 Support-Stratum Generator Reset` 的真实复盘制定。  
> v9.2.61 的 terminal route 是：
>
> ```text
> route = R13-ValueRiskOrthogonalizationFail
> base_candidate = LQ-t2-h256
> success_v9261_strict_purekan_functional = False
> success_v9261_full_functional = False
> success_v9261_external_ready = False
> ```
>
> v9.2.61 的关键事实是：
>
> ```text
> P1 three-class decomposition:
>   pass = 1
>   accepted count = 494
>   safe-good accepted = 368
>   harmless-null accepted = 122
>   bad-event accepted = 4
>   useful missed = 5389
>
> P2 support-stratum generator reset:
>   pass = 1
>   natural rows = 24192
>   balanced diagnostic rows = 6000
>   signal strata = 16
>   families = 829
>   duplicate rows = 0
>
> P3 value/control statistic:
>   best non-reference = VAL3-NoRegretControlMargin
>   useful AUC = 0.998329
>   value_stat_pass = 1
>   value gate coverage = 0.0078125
>   utility pass = 0
>
> P4 risk-null aware statistic:
>   best = RNULL5-ThreeClassSoftmaxMargin
>   bad-event AUC = 0.749987
>   null AUC = 0.471273
>   coverage = 0.009673
>   risk_null_stat_pass = 0
>
> P5 support frontier:
>   best = SUP3-ValueRiskKNNPocket
>   precision = 0.841398
>   bad-event = 0.024194
>   accepted strata = 3
>   coverage = 0.015377 < 0.03
>   official pass = 0
>
> P6 exact reference deployable frontier:
>   best = C-VAL2-ControlGapLCB + RNULL3-RiskValueJointBudget + SUP3-ValueRiskKNNPocket
>   precision = 0.750600
>   bad-event = 0.045564
>   coverage = 0.017237 < 0.03
>   precision LCB = 0.706911 < 0.75
>   bad-event UCB = 0.070063 > 0.05
>   exact_reference_deployable = 0
>
> P7 true-delta compute:
>   AUC = 0.879072
>   agreement = 1.0
>   step ratio q90 = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> current blocker:
>   value_control_statistic_failed
>
> next_required_implementation:
>   reset_value_control_target_decomposition
> ```
>
> v9.2.62 的核心判断是：
>
> $$
> \boxed{
> \text{support 生成器已经过关，risk 可以压 bad-event，value ranking 极强；但 current value/control gate 仍不能给出可部署 coverage。}
> }
> $$
>
> 这不是 functional update 失败，也不是 exact branch-delta signal 消失。相反，v9.2.61 第一次同时证明了：
>
> $$
> \boxed{
> \text{support-stratum generator 可以真实扩展到 16 strata / 829 families。}
> }
> $$
>
> $$
> \boxed{
> \text{useful signal 可被 legal value/control statistic 几乎完美排序。}
> }
> $$
>
> $$
> \boxed{
> \text{bad-event 可被 risk statistic 预测，但 harmless-null 仍没有被可靠分离。}
> }
> $$
>
> 因此 v9.2.62 不能继续做：
>
> ```text
> 再调一个 support threshold；
> 再硬化 risk gate；
> 再扩 balanced rows；
> 再写一个 broad support pocket；
> 再继续纯 kernelization；
> 按 dataset 单独调 controller。
> ```
>
> 本轮必须做的是：
>
> $$
> \boxed{
> \text{重设 useful/control/null 的目标分解，专门解决 high-AUC value score 只能 tiny coverage 的问题。}
> }
> $$

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
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件。允许诊断：

```text
MNIST / Fashion-MNIST / KMNIST 的 useful/null/bad-event 分布；
不同 dataset 的 value miss / control-equivalent / tail-risk 模式；
不同 dataset 的 support family scarcity；
leave-dataset-out failure matrix；
per-dataset visualization。
```

禁止：

```text
if dataset == Fashion: use value threshold A
if dataset == KMNIST: use null rejector B
if dataset == MNIST: skip exact delta
按 dataset 调 threshold
按 dataset 选择 controller
按 dataset 选择 support family
使用 validation/test metric at commit time
使用 posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-support-family-level / branch-delta features。Dataset name 只能进入 report，不得进入 commit rule。

---

# Part I. 对 v9.2.61 的独立判断

## 1. v9.2.61 没有达到目标

v9.2.61 没有达到 strict PureKAN functional success。原因不是 full-run 输了，而是更前面的 gates 没有闭合：

```text
value utility pass = 0
risk_null_stat_pass = 0
support_frontier_pass = 0
exact_reference_deployable = 0
true_delta_compute_pass = 0
LDO / LSO / paired replay = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

P6 的 best exact-reference point estimate 已接近可部署区间：

```text
precision = 0.750600
bad-event = 0.045564
coverage = 0.017237
```

但它仍不能 official，因为 coverage 低于 `0.03`，precision LCB 低于 `0.75`，bad-event UCB 高于 `0.05`。这不是可忽略的小误差，而是说明 accepted region 的统计置信度与覆盖率都不够。

## 2. v9.2.61 的真实进展

v9.2.61 是近几轮中最重要的推进之一。

第一，support-stratum generator 终于过关。v9.2.60 只有 `591` balanced diagnostic rows 和 `3` measured signal strata；v9.2.61 达到：

```text
natural rows = 24192
balanced diagnostic rows = 6000
signal strata = 16
families = 829
duplicate rows = 0
```

这意味着过去长期存在的 “support measurement too narrow” blocker 至少在 measurement 层面被实质推进。现在不能再简单说 controller 失败是因为只有 3 个 strata。

第二，three-class decomposition 闭合。accepted rows 中：

```text
safe-good = 368
harmless-null = 122
bad-event = 4
```

这说明当前 accepted region 的主要污染不是 bad-event，而是 harmless-null。也就是说，risk 已经相当有效地压住了 bad-event，但 accepted rows 里还有很多安全但没有 useful functional value 的样本。

第三，value/control signal 很强。`VAL3-NoRegretControlMargin` 的 useful AUC 达到 `0.998329`。这不是弱信号，而是强到足以说明 “useful vs non-useful” 在某种 legal feature 空间里是可排序的。但当前 gate 只能给出 `0.0078125` coverage，说明问题在 **target decomposition / threshold geometry / confidence calibration**，不是 useful signal 完全不存在。

第四，support frontier 有诊断推进。`SUP3-ValueRiskKNNPocket` 达到：

```text
precision = 0.841398
bad-event = 0.024194
accepted strata = 3
coverage = 0.015377
```

这个结果比 v9.2.60 的 single-stratum tiny pocket 更有意义：它已经满足 precision、bad-event、multi-strata 的直觉方向，只是 coverage 仍差约一半。

## 3. v9.2.61 的真实失败

v9.2.61 的失败不是 “risk/support 完全失败”，而是：

$$
\boxed{
\text{useful / harmless-null / bad-event 三类分离仍然没有形成 coverage-preserving controller。}
}
$$

具体有三层。

### 3.1 Value/control ranking 很强，但 gate coverage 低

`useful AUC = 0.998329` 与 `value gate coverage = 0.0078125` 同时出现，说明一个微妙问题：排序强，但可部署区域很窄。这通常意味着 target 或 policy 把 “有用” 定义得过尖锐，或者 threshold 只吃到了极端高置信区，无法把中等置信但实际可用的 rows 纳入。

### 3.2 Risk-null 分离没闭合

P4 的 bad-event AUC 是 `0.749987`，但 null AUC 只有 `0.471273`。这说明 risk score 可以识别 “坏”，但不能识别 “安全但没用”。如果 harmless-null 不被识别，controller 会出现两类坏结果：

```text
宽 gate:
  coverage 上升，但 harmless-null 大量混入，precision 不稳。

窄 gate:
  precision 和 bad-event 好看，但 coverage 低于 0.03。
```

### 3.3 Exact-reference frontier 已接近但不可信

P6 point estimate：

```text
precision = 0.750600
bad-event = 0.045564
coverage = 0.017237
```

这说明方向不是错的。但：

```text
coverage < 0.03
precision LCB = 0.706911
bad-event UCB = 0.070063
```

这说明它没有统计安全余量。一个真正可部署 controller 不能只在点估计上刚好碰线；它必须在 bootstrap / confidence bound 下也稳。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{useful-control target decomposition and null rejection are still under-specified.}
}
$$

更具体地说：

```text
1. support generator 已经不再是主 blocker；
2. bad-event risk 已有可用信号；
3. useful ranking 极强；
4. exact-reference frontier 接近 point-estimate gate；
5. 但 harmless-null 分离弱，coverage 不足，confidence bound 不足；
6. true-delta compute 仍太贵，但 decision geometry 还没资格打开 official controller。
```

所以 v9.2.62 的重点不是扩更多 support，也不是重做 risk。它要回答：

```text
为什么 useful AUC 接近 1，却只能得到 0.0078125 的 coverage？
harmless-null 到底是哪几种 null？
能否把 useful/control target 分解成多种可部署 submodes？
能否把 exact reference coverage 从 0.017237 推到 >=0.03，同时让 LCB/UCB 过线？
```

---

# Part II. v9.2.62 总体目标

v9.2.62 的总体目标是：

$$
\boxed{
\text{通过 useful-control target reset 与 null-submode rejection，把 exact-reference frontier 推入 deployable region。}
}
$$

本轮必须并行验证五件事：

```text
1. useful target 是否被过窄定义或过尖锐校准；
2. harmless-null 是否可分解为 control-equivalent / low-gain / low-horizon-persistence / support-only / delta-silent；
3. useful gate 是否能从 high-confidence tiny slice 扩展到 mid-confidence useful region；
4. exact-reference frontier 是否能通过 LCB/UCB 和 coverage gate；
5. true-delta compute 是否继续向 system envelope 收敛。
```

本轮最重要的 stop-go 不是 “某个 AUC 是否更高”，而是：

$$
\boxed{
\text{exact reference deployable frontier 是否恢复。}
}
$$

如果 exact reference deployable 仍失败，kernelization 不是主线；如果 reference feasible 但 true-delta compute 仍失败，kernelization 才重新成为主 blocker。

---

# Part III. 核心假设

## H1：v9.2.61 的 value/control failure 是 target decomposition failure，不是 useful signal absence

证据：

```text
useful AUC = 0.998329
value_stat_pass = 1
value gate coverage = 0.0078125
utility pass = 0
```

H1 成立标准：

经过 useful-control target reset 后，至少一个 value-control controller 达到：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\geq0.03,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

并且：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

H1 失败标准：

所有 target reset 后的 useful/control gates 仍：

```text
coverage < 0.02
```

或一扩 coverage 就：

```text
precision < 0.75
bad-event > 0.05
```

## H2：harmless-null 是多模式混合，不能用单一 null score 分开

P4 的 null AUC 只有 `0.471273`，但三类 decomposition 已经证明 harmless-null 存在。H2 认为 null 不是一个单一类别，而是多个不同原因导致的 “not useful”：

```text
N1-control-equivalent:
  RealFunctional 不明显强于 AdamWParallel / bestLR。

N2-low-real-gain:
  RealFunctional 自身 gain 不足。

N3-low-horizon-persistence:
  短 horizon 有信号，长 horizon 消失。

N4-delta-silent:
  delta norm / logit displacement 不足。

N5-support-only:
  support 稳定但 value 不足。

N6-risk-safe-null:
  risk 很低，但 value/control 也低。
```

H2 成立标准：

至少三个 null submodes 有 legal statistic 达到：

$$
AUC_{\text{null-submode}}\geq0.65
$$

或：

$$
Corr_{\text{null-submode}}\geq0.30.
$$

组合 null rejector 达到：

$$
NullRate_{\text{accepted}}\leq0.15
$$

并保持：

$$
Coverage\geq0.03.
$$

H2 失败标准：

所有 null submode statistic 都低于 diagnostic threshold，或 null rejector 降低 null 但把 coverage 压回 tiny slice。

## H3：support generator 已经足够，下一步是 use support for stability not discovery

v9.2.61 已经达到：

```text
balanced diagnostic rows = 6000
signal strata = 16
families = 829
```

H3 认为本轮 support 不再是 discovery blocker，而是 confidence / stability blocker。

H3 成立标准：

P2 support audit 保持：

```text
natural_real_event_count >= 24000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 12
measured_family_count >= 600
duplicate_row_count = 0
```

并且 exact-reference accepted set 满足：

```text
accepted_signal_strata_count >= 3
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

H3 失败标准：

support generator 回退，或 accepted set 仍被单一 pocket 支配。

## H4：P6 point estimate 接近 gate，coverage/confidence 可以通过 calibration 而非新 primitive 修复

v9.2.61 P6：

```text
precision = 0.750600
bad-event = 0.045564
coverage = 0.017237
precision LCB = 0.706911
bad-event UCB = 0.070063
```

H4 成立标准：

通过 threshold smoothing、risk-budget calibration、mid-confidence useful extension 后：

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

H4 失败标准：

point estimates 可以过线，但 LCB/UCB 始终不过，说明 frontier 不稳定。

## H5：true-delta compute 仍是并行 blocker，不是当前唯一主 blocker

P7 仍是：

```text
AUC = 0.879072
agreement = 1.0
step ratio = 2.863280
```

H5 成立标准：

至少一个 true-delta compute candidate 达到：

$$
AUC\geq0.70,
$$

$$
Agreement\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H5 失败标准：

true-delta compute 仍太贵或 signal lost。  
但如果 H1-H4 失败，即使 H5 成立，也不能进入 functional success。

---

# Part IV. New target decomposition

## 1. Useful 不再是单一标签

v9.2.62 把 useful 拆成：

$$
Y_{\text{useful}}
=
Y_{\text{real-gain}}
\land
Y_{\text{control-resistant}}
\land
Y_{\text{persistent}}
\land
Y_{\text{efficient}}
\land
Y_{\text{not-null}}.
$$

其中：

$$
Y_{\text{real-gain}}(e)
=
\mathbb{1}
[
Gain_F(e) > \gamma_F
].
$$

$$
Y_{\text{control-resistant}}(e)
=
\mathbb{1}
[
Gain_F(e)
-
\max(Gain_{\text{AdamWParallel}}(e),Gain_{\text{bestLR}}(e))
>
\gamma_C
].
$$

$$
Y_{\text{persistent}}(e)
=
\mathbb{1}
[
\min_{h\in\mathcal{H}}
Gap_h(e)
>
\gamma_H
].
$$

$$
Y_{\text{efficient}}(e)
=
\mathbb{1}
[
\frac{Gain_F(e)-\max(Gain_{\text{controls}}(e))}
{\|\Delta z_F(e)\|+\epsilon}
>
\gamma_E
].
$$

$$
Y_{\text{not-null}}(e)
=
1-Y_{\text{harmless-null}}(e).
$$

## 2. Harmless-null submodes

定义：

$$
Y_{\text{null}}
=
Y_{N1}\lor Y_{N2}\lor Y_{N3}\lor Y_{N4}\lor Y_{N5}\lor Y_{N6}.
$$

其中：

$$
Y_{N1}=
\mathbb{1}
[
Gain_F-\max(Gain_{\text{controls}})\leq\gamma_C
].
$$

$$
Y_{N2}=
\mathbb{1}
[
Gain_F\leq\gamma_F
].
$$

$$
Y_{N3}=
\mathbb{1}
[
\min_h Gap_h\leq\gamma_H
].
$$

$$
Y_{N4}=
\mathbb{1}
[
\|\Delta z_F\|\leq\epsilon_z
].
$$

$$
Y_{N5}=
\mathbb{1}
[
SupportStable=1
]
\land
\mathbb{1}
[
Y_{\text{real-gain}}=0
].
$$

$$
Y_{N6}=
\mathbb{1}
[
P_{\text{bad}}\leq\rho
]
\land
\mathbb{1}
[
Y_{\text{useful}}=0
].
$$

## 3. Controller decomposition

Official controller form:

$$
Accept(e)
=
UsefulPositive(e)
\land
NullRejected(e)
\land
RiskSafe(e)
\land
SupportStable(e).
$$

where:

$$
UsefulPositive(e)
=
\mathbb{1}
[
S_{\text{useful}}(e)\geq\tau_U
],
$$

$$
NullRejected(e)
=
\mathbb{1}
[
P_{\text{null}}(e)\leq\rho_N
],
$$

$$
RiskSafe(e)
=
\mathbb{1}
[
P_{\text{bad}}(e)\leq\rho_B
],
$$

$$
SupportStable(e)
=
\mathbb{1}
[
Rel(family(e))\geq r_0
]
\cdot
\mathbb{1}
[
Density(e)\geq d_0
].
$$

Calibration objective:

$$
\max Coverage(A)
$$

subject to:

$$
Precision_{\text{LCB}}(A)\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}(A)\leq0.05,
$$

$$
Coverage(A)\in[0.03,0.15],
$$

$$
NullRate(A)\leq0.15,
$$

$$
max\_family\_share(A)\leq0.50,
$$

$$
accepted\_signal\_strata\_count(A)\geq3.
$$

---

# Part V. Candidate statistics

## 1. Useful-control statistics

### U0：v9.2.61 reference useful score

Reference only.

### U1：RealGainLCBv2

$$
LCB_{\text{RealGain}}(f)
=
\mu_{Gain_F}(f)
-
\kappa
\sqrt{
\frac{\sigma^2_{Gain_F}(f)}{n_f+\epsilon}
}.
$$

### U2：ControlGapLCBv2

$$
LCB_{\text{Gap}}(f)
=
\mu_{Gain_F-\max(Gain_{\text{controls}})}(f)
-
\kappa
\sqrt{
\frac{\sigma^2_{Gap}(f)}{n_f+\epsilon}
}.
$$

### U3：PersistentControlGap

$$
S_{\text{persist}}
=
\min_{h\in\{20,80,240,640\}}
Gap_h.
$$

If only a subset of horizons is available at commit time, use the train-stream accumulated horizon windows already available before commit; do not use validation/test.

### U4：DeltaEfficiencyControl

$$
S_{\text{eff}}
=
\frac{
Gain_F-\max(Gain_{\text{controls}})
}{
\|\Delta z_F\|+\epsilon
}.
$$

### U5：NoRegretControlMarginV2

Extend VAL3 by adding stronger controls:

```text
AdamWParallel
bestLR
NoOp
RandomMatched
LRScaled
TrustRatioScaled
```

$$
S_{\text{noregret}}
=
Gain_F-\max_{c\in\mathcal{C}} Gain_c.
$$

### U6：UsefulMixtureScore

Monotone score:

$$
S_{\text{useful}}
=
a_1LCB_{\text{RealGain}}
+
a_2LCB_{\text{Gap}}
+
a_3S_{\text{persist}}
+
a_4S_{\text{eff}}
+
a_5S_{\text{noregret}}.
$$

Coefficients are selected on calibration split only, no dataset name.

---

## 2. Null-submode rejectors

### N0：v9.2.61 null reference

Reference only.

### N1：ControlEquivalentNullRejector

$$
P_{N1}(e)
=
\sigma(
-\alpha
[
Gain_F-\max(Gain_{\text{controls}})
]
).
$$

### N2：LowGainNullRejector

$$
P_{N2}(e)=\sigma(-\alpha Gain_F).
$$

### N3：NonPersistentNullRejector

$$
P_{N3}(e)
=
\sigma(
-\alpha \min_h Gap_h
).
$$

### N4：DeltaSilentNullRejector

$$
P_{N4}(e)
=
\sigma(
-\alpha
\frac{\|\Delta z_F\|}{\operatorname{median}(\|\Delta z_F\|)+\epsilon}
).
$$

### N5：SupportOnlyNullRejector

$$
P_{N5}(e)
=
\mathbb{1}[SupportStable(e)=1]
\cdot
\sigma(-\alpha S_{\text{useful}}(e)).
$$

### N6：HybridNullProbability

$$
P_{\text{null}}(e)
=
1-\prod_{m=1}^{5}(1-P_{Nm}(e)).
$$

or calibrated softmax triage:

$$
P_{\text{safe-good}},P_{\text{null}},P_{\text{bad}}
=
\operatorname{softmax}
(
S_{\text{useful}},
S_{\text{null}},
S_{\text{bad}}
).
$$

---

## 3. Risk statistics

### R0：v9.2.61 RNULL reference

Reference only.

### R1：TailFirstRiskBudgetV3

Tail risk remains a hard-ish veto:

$$
RiskSafe(e)
=
\mathbb{1}[p_{\text{tail}}(e)\leq\rho_t]
\land
\mathbb{1}
[
\lambda_m p_{\text{mismatch}}
+
\lambda_c p_{\text{control}}
+
\lambda_f p_{\text{family}}
\leq B_0
].
$$

### R2：UsefulConditionedBadRisk

Bad-event threshold is conditioned on useful score:

$$
\rho_B(e)
=
\rho_0
+
\alpha
\cdot
\sigma(S_{\text{useful}}(e)-\tau_U).
$$

Accept only if:

$$
P_{\text{bad}}(e)\leq\rho_B(e).
$$

### R3：BadNullJointBudget

$$
Budget(e)
=
\lambda_B P_{\text{bad}}(e)
+
\lambda_N P_{\text{null}}(e)
-
\lambda_U S_{\text{useful}}(e).
$$

Gate:

$$
Budget(e)\leq B_0.
$$

---

## 4. Support statistics

### S0：v9.2.61 SUP3 reference

Reference only.

### S1：UsefulRiskKNNPocketV2

Feature vector:

```text
useful score
control gap
null probability
bad probability
tail risk
delta efficiency
support density
family reliability
horizon bucket
stratum bucket
role bucket
```

Density:

$$
Density(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### S2：MultiResolutionUsefulFamily

Family:

$$
family(e)
=
(
stratum,
horizon,
role\_bucket,
carrier,
control\_gap\_bucket,
useful\_bucket,
null\_bucket,
bad\_risk\_bucket,
delta\_efficiency\_bucket
).
$$

Empirical-Bayes reliability:

$$
Rel_{\text{EB}}(f)
=
\frac{n_f}{n_f+\alpha}\hat{p}_f
+
\frac{\alpha}{n_f+\alpha}\hat{p}_{parent(f)}.
$$

### S3：LeaveUsefulFamilyOutReliability

$$
Rel_{\text{LUFO}}(f)
=
\min_{f'\in\mathcal{N}(f)}
Rel_{\text{EB}}(f').
$$

### S4：CoverageBalancedSupportCap

Accepted set must satisfy:

```text
accepted_signal_strata_count >= 3
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

---

# Part VI. Controller candidates

## C0：v9.2.61 best reference

Reference only.

## C1：UsefulFirstNullRejectController

$$
Accept(e)
=
S_{\text{useful}}(e)\geq\tau_U
\land
P_{\text{null}}(e)\leq\rho_N
\land
P_{\text{bad}}(e)\leq\rho_B
\land
SupportStable(e).
$$

## C2：RB3PlusUsefulExpansion

Starts from v9.2.60 / v9.2.61 safety-feasible risk region, then adds value-control gate:

$$
Accept(e)
=
RiskSafe(e)
\land
S_{\text{useful}}(e)\geq\tau_U
\land
P_{\text{null}}(e)\leq\rho_N
\land
SupportStable(e).
$$

## C3：TriageMarginController

Explicitly separates safe-good from null and bad:

$$
Accept(e)
=
\mathbb{1}
[
P_{\text{safe-good}}(e)
-
\max(P_{\text{null}}(e),P_{\text{bad}}(e))
\geq \tau_T
].
$$

## C4：PersistentUsefulController

Requires medium useful score plus horizon persistence:

$$
Accept(e)
=
S_{\text{useful}}(e)\geq\tau_U
\land
S_{\text{persist}}(e)\geq\tau_H
\land
RiskSafe(e)
\land
SupportStable(e).
$$

## C5：ConstrainedCoverageOptimizerV3

Search thresholds to maximize coverage:

$$
\max Coverage(A)
$$

subject to:

$$
Precision_{\text{LCB}}(A)\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}(A)\leq0.05,
$$

$$
Coverage(A)\in[0.03,0.15],
$$

$$
NullRate(A)\leq0.15,
$$

$$
accepted\_signal\_strata\_count(A)\geq3,
$$

$$
max\_family\_share(A)\leq0.50.
$$

## C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.61 boundary reproduction

### 目标

确认最新 boundary 稳定，避免基于偶然 run 制定 route。

### 必须记录

```text
route
source_route_v9260
three_class_decomposition_pass
support_stratum_generator_pass
value_stat_pass
value_gate_coverage
risk_null_stat_pass
null_auc
support_frontier_precision
support_frontier_coverage
support_frontier_bad_event
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_precision_lcb
reference_bad_event_ucb
true_delta_compute_pass
true_delta_step_ratio
oracle_support_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R13-ValueRiskOrthogonalizationFail
support generator pass = 1
value/control AUC strong
risk-null pass = 0
exact_reference_deployable = 0
oracle_support_pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_v9260_to_v9261_progress_ladder.svg
p0_value_risk_support_gate_ladder.svg
```

---

## P1：useful-control target autopsy

### 目标

解释为什么 useful AUC 很高但 value gate coverage 很低。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
safe_good
useful
harmless_null
bad_event
Y_real_gain
Y_control_resistant
Y_persistent
Y_efficient
Y_not_null
S_VAL3
S_real_gain_lcb
S_control_gap_lcb
S_persistent
S_delta_efficiency
accepted_by_reference
missed_useful
value_gate_rejected_reason
```

Rejected reasons：

```text
V1-real-gain-low
V2-control-gap-low
V3-horizon-not-persistent
V4-delta-efficiency-low
V5-null-prob-high
V6-support-unstable
V7-confidence-bound-too-low
```

### 判断标准

P1 pass：

```text
useful_rejected_attribution_fraction >= 0.90
harmless_null_attribution_fraction >= 0.90
bad_event_attribution_fraction >= 0.90
at least two useful submodes have positive density >= 0.03
```

### 可视化

```text
p1_useful_rejected_sankey.svg
p1_useful_score_vs_coverage.svg
p1_value_subfactor_venn.svg
p1_harmless_null_mode_breakdown.svg
```

---

## P2：null-submode decomposition and rejector factory

### 目标

把 harmless-null 拆成可预测 submodes，修复 null AUC 低的问题。

### 必须记录

```text
null_rejector_id
null_submode
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_null_submode
corr_null_submode
P_null_calibration_error
null_rate_after_gate
safe_good_precision_after_gate
coverage_after_gate
bad_event_after_gate
feature_overhead
memory_overhead
```

### 判断标准

Null submode pass：

At least three null submodes satisfy:

$$
AUC_{\text{null-submode}}\geq0.65
$$

or:

$$
Corr_{\text{null-submode}}\geq0.30.
$$

Null rejector utility pass：

$$
NullRate_{\text{accepted}}\leq0.15,
$$

$$
Precision_{\text{safe-good}}\geq0.75,
$$

$$
Coverage\geq0.03,
$$

$$
BadEventRate\leq0.05.
$$

Diagnostic pass：

$$
NullRate_{\text{accepted}}\leq0.25,
$$

$$
Coverage\geq0.02.
$$

### 可视化

```text
p2_null_submode_auc_matrix.svg
p2_null_rejection_frontier.svg
p2_safe_good_null_bad_triage.svg
p2_null_feature_ablation.svg
```

---

## P3：useful-control statistic reset

### 目标

从 high-AUC tiny gate 转向 coverage-preserving useful gate。

### 必须记录

```text
useful_stat_id
features_used
subfactors_used
AUC_useful
AUC_real_gain
AUC_control_resistant
AUC_persistent
AUC_efficiency
corr_useful
precision_after_useful_gate
coverage_after_useful_gate
bad_event_after_useful_gate
null_rate_after_useful_gate
precision_lcb
bad_event_ucb
```

### 判断标准

Useful statistic pass：

$$
AUC_{\text{useful}}\geq0.70
$$

or:

$$
Corr_{\text{useful}}\geq0.35.
$$

Utility pass：

$$
Precision_{\text{safe-good}}\geq0.75,
$$

$$
Coverage\geq0.03,
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15.
$$

Diagnostic pass：

$$
Coverage\geq0.02
$$

and:

$$
Precision_{\text{safe-good}}\geq0.70.
$$

### 可视化

```text
p3_useful_control_auc_matrix.svg
p3_useful_gate_precision_coverage_bad_null.svg
p3_control_gap_lcb_curve.svg
p3_persistence_efficiency_ablation.svg
```

---

## P4：support stability audit after generator pass

### 目标

确认 support generator pass 没有退化，并把 support 用于 confidence，而不是重新做 discovery。

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family
family_definition
support_density
family_reliability
leave_family_out_reliability
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
duplicate_row_flag
source_hash
```

### 判断标准

Support stability pass：

```text
natural_real_event_count >= 24000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 12
measured_family_count >= 600
duplicate_row_count = 0
```

Accepted support pass：

```text
accepted_signal_strata_count >= 3
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p4_signal_strata_coverage.svg
p4_family_reliability_heatmap.svg
p4_accepted_family_balance.svg
p4_support_confidence_intervals.svg
```

---

## P5：exact reference deployable frontier v6

### 目标

组合 P2/P3/P4，在 CBD0 exact reference 上判断 deployable accept frontier 是否恢复。

### 必须记录

```text
controller_id
useful_stat_id
null_rejector_id
risk_stat_id
support_stat_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
legal_oracle_jaccard
dataset_name_used
posthoc_used_at_commit
reference_only
```

### 判断标准

Exact reference deployable pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
NullRate_{\text{heldout}}\leq0.15.
$$

Confidence pass：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

Support pass：

```text
accepted_signal_strata_count >= 3
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

If no controller passes, route must be:

```text
R10-ExactReferenceStillNotDeployableAfterUsefulReset
```

### 可视化

```text
p5_reference_frontier_v6_precision_coverage_bad_null.svg
p5_lcb_ucb_frontier.svg
p5_useful_null_risk_threshold_surface.svg
p5_oracle_legal_gap_after_useful_reset.svg
```

---

## P6：true-delta compute v7 parallel lane

### 目标

继续推进 system path，但不让 compute lane 掩盖 decision feasibility。

### 必须记录

```text
custom_delta_id
layout
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
AUC_safe_good
AUC_useful
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
value_component_time
risk_null_component_time
support_component_time
kernel_count
sync_count
read_MB
write_MB
```

### 判断标准

Compute pass：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic compute pass：

$$
StepRatio_{q90}\leq2.00
$$

with signal retained.

### 可视化

```text
p6_true_delta_v7_cost_signal_pareto.svg
p6_compute_vs_decision_ladder.svg
p6_residual_subphase_after_useful_reset.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P5 reference deployable pass 与 P6 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
useful_stat_id
null_rejector_id
risk_stat_id
support_stat_id
thresholds
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Official controller pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
NullRate_{\text{heldout}}\leq0.15,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p7_system_controller_precision_coverage_bad_null.svg
p7_system_controller_cost_vs_value.svg
p7_system_controller_family_coverage.svg
```

---

## P8：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
controller_id
custom_delta_id
useful_stat_id
null_rejector_id
risk_stat_id
support_stat_id
threshold
precision
coverage
bad_event_rate
null_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_hidden_dataset_tuning_audit.svg
p8_leaveout_failure_modes.svg
```

---

## P9：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P8 survivor
custom_delta_id = best P8 survivor
useful_stat_id = best P8 survivor
null_rejector_id = best P8 survivor
risk_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledUsefulStat, ShuffledNullRejector,
           ShuffledRiskStat, ShuffledSupportStat, ShuffledControlGain,
           ShuffledCandidateGate, ShuffledBranchRatio, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
useful_stat_id
null_rejector_id
risk_stat_id
support_stat_id
dataset
seed
horizon
signal_stratum
event_family
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
null_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

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

Shuffle controls fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledUsefulStat = fail
ShuffledNullRejector = fail
ShuffledRiskStat = fail
ShuffledSupportStat = fail
ShuffledControlGain = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p9_official_paired_replay_pareto.svg
p9_macro_beat_rate.svg
p9_signal_stratum_win_matrix.svg
p9_shuffle_control_matrix.svg
p9_system_gate_distribution.svg
```

---

## P10：Short-run scout

### 目标

如果 P9 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledTrueBranchDelta
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
null_rate
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

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

# Part VIII. Required artifacts

```text
run_manifest.json
contract_audit_v9262.csv
p0_v9261_boundary_reproduction.csv
p1_useful_control_target_autopsy.csv
p2_null_submode_decomposition_and_rejector_factory.csv
p3_useful_control_statistic_reset.csv
p4_support_stability_audit_after_generator_pass.csv
p5_exact_reference_deployable_frontier_v6.csv
p6_true_delta_compute_v7_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
useful_target_trace_v9262.csv
null_submode_trace_v9262.csv
useful_control_stat_trace_v9262.csv
support_stability_trace_v9262.csv
reference_frontier_v6_trace.csv
true_delta_compute_v7_trace.csv
system_controller_trace_v9262.csv
leaveout_trace_v9262.csv
paired_replay_branch_trace_v9262.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9261_boundary_unstable
F3_dataset_tuning_detected
F4_useful_target_autopsy_incomplete
F5_null_submode_unattributed
F6_null_rejector_not_predictive
F7_null_rejector_tiny_coverage
F8_useful_stat_tiny_coverage
F9_useful_stat_precision_fail
F10_support_generator_regression
F11_support_stability_fail
F12_exact_reference_still_not_deployable_after_useful_reset
F13_precision_lcb_fail
F14_bad_event_ucb_fail
F15_true_delta_compute_still_expensive
F16_true_delta_signal_lost
F17_system_controller_precision_fail
F18_system_controller_coverage_fail
F19_system_controller_bad_event_fail
F20_system_controller_null_rate_fail
F21_leave_dataset_out_fail
F22_leave_stratum_out_fail
F23_paired_replay_control_equivalent
F24_shuffle_control_pass
F25_functional_lr_equivalent
F26_short_run_task_drop
F27_full_run_no_macro_hard_stratum_gain
F28_strong_baseline_explains_gain
F29_robustness_fail
F30_external_not_ready
F31_fake_or_proxy_violation
F32_artifact_missing
```

---

# Part IX. Route decision

```text
R1-BoundaryReproduced:
  v9.2.61 boundary is reproduced.

R2-UsefulTargetAutopsyPass:
  useful rejection and harmless-null modes are attributed.

R3-NullSubmodeRejectorPass:
  harmless-null submodes have legal predictive rejectors.

R4-UsefulControlStatisticPass:
  useful/control statistic becomes coverage-preserving.

R5-SupportStabilityPass:
  support generator remains broad and accepted set is balanced.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + useful/null/risk/support frontier passes precision / coverage / bad-event / null gate.

R7-TrueDeltaComputeV7SystemPass:
  true branch-delta v7 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-UsefulTargetStillTiny:
  useful/control statistic still only produces tiny coverage.

R13-NullRejectorFail:
  harmless-null remains inseparable under legal features.

R14-ReferenceFeasibleButComputeFail:
  decision geometry viable, true-delta compute still too expensive.

R15-ComputePassButReferenceFail:
  compute viable, accept-region geometry still fails.

R16-SupportGeneratorRegression:
  support generator no longer satisfies broad support.

R17-ExactReferenceStillNotDeployableAfterUsefulReset:
  even after useful/null reset, exact reference cannot form deployable frontier.

R18-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R19-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R20-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R21-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9261_boundary_pass
dataset_tuning_detected
useful_target_autopsy_pass
useful_rejected_attribution_fraction
harmless_null_attribution_fraction
bad_event_attribution_fraction
best_null_rejector_id
null_rejector_pass
null_auc
null_rate_after_gate
best_useful_stat_id
useful_stat_pass
useful_auc
useful_precision_after_gate
useful_coverage_after_gate
useful_bad_event_after_gate
support_stability_pass
natural_real_event_count
balanced_diagnostic_real_event_count
measured_signal_strata_count
measured_family_count
duplicate_row_count
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
best_reference_controller_id
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
best_true_delta_id
true_delta_compute_pass
true_delta_auc
true_delta_agreement
true_delta_step_ratio_q90
true_delta_memory_ratio
best_system_controller_id
system_legal_controller_pass
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9262_strict_purekan_functional
success_v9262_full_functional
success_v9262_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 useful-control target autopsy
  P2 null-submode rejector factory
  P3 useful-control statistic reset
  P4 support stability audit
  P5 exact reference deployable frontier v6
  P6 true-delta compute v7

Batch 2:
  P7 system-legal controller calibration
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if paired replay passes

Batch 4:
  full 10-seed / robustness / strong baseline only if P10 passes
```

Gate rule：

```text
P5 exact reference deployable frontier can pass diagnostically before compute pass.
P7 official controller cannot pass unless P5 reference deployable frontier and P6 compute pass.
P8/P9 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  system-legal controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.61 boundary reproduced
useful target autopsy completed
null submode decomposition measured
useful-control statistic reset measured
support stability audited
exact reference deployable frontier v6 measured
true-delta compute v7 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
exact reference deployable frontier pass
+
useful/null/risk/support stats pass
+
multi-stratum / multi-family support pass
```

## Compute success

```text
Minimum diagnostic success
+
at least one true branch-delta path predicts safe-good
+
accept agreement pass
+
system overhead gate pass
```

## Legal controller success

```text
Decision geometry success
+
Compute success
+
system-legal true-delta controller heldout pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
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

## Failure stop

```text
1. v9.2.61 boundary cannot be reproduced；
2. useful-control target autopsy cannot attribute misses/nulls；
3. harmless-null remains inseparable；
4. useful statistic remains high-AUC but tiny-coverage；
5. support generator regresses；
6. exact reference still not deployable after useful/null reset；
7. precision LCB remains below 0.75；
8. bad-event UCB remains above 0.05；
9. true branch-delta compute remains too expensive；
10. true branch-delta compute passes system but loses signal；
11. system-legal controller cannot meet precision / coverage / bad-event / null-rate；
12. leave-dataset-out fails；
13. leave-stratum-out fails；
14. paired replay remains control-equivalent；
15. shuffle controls pass, indicating overfit；
16. short-run task drops；
17. full run gives no macro / hard-stratum / geometry gain；
18. functional breaks system gate；
19. gains are explained by QuadraticFeatureMLP；
20. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：reference deployable + compute + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：reference deployable frontier pass but compute fail

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused useful-null-risk-support compute，不调 dataset。

## Case C：compute pass but reference frontier fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 useful/null/risk/support sufficient statistics，不走 kernel-only。

## Case D：useful-control target reset fails

必须声明：

```text
useful signal ranks well but cannot produce coverage-preserving legal accept region.
```

下一步重设 useful/control target or output-delta sufficient statistics。

## Case E：null rejection fails

必须声明：

```text
bad-event is controllable, but harmless-null remains inseparable from useful safe-good.
```

下一步必须重设 null decomposition，不继续 hard-risk gating。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism。

---

# Part XIII. 最终建议

v9.2.62 的一句话策略是：

$$
\boxed{
\text{不要再扩 support，也不要再硬化 risk；现在要重设 useful/control/null target，让 high-AUC value signal 变成 coverage-preserving controller。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
TBD0 有没有 AUC；
support generator 是否能过；
K7d local fusion 是否能跑；
formula proxy 是否够快；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 为什么 useful AUC = 0.998329，却只能得到 coverage 0.0078125？
2. harmless-null 是哪些 submodes？
3. null AUC 低是因为 target 混合，还是 legal feature 缺失？
4. exact-reference frontier 能否把 coverage 从 0.017237 拉到 >=0.03？
5. precision LCB 能否从 0.706911 拉到 >=0.75？
6. bad-event UCB 能否从 0.070063 压到 <=0.05？
7. support generator pass 后，accepted set 能否保持 >=3 strata / >=32 families？
8. true-delta compute v7 能否继续压到 step <=1.50？
9. system-legal controller 能否 LDO/LSO？
10. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.62 的结果将给出清晰分叉：

```text
if useful/null reset restores reference deployability and true-delta compute passes:
  open system-legal controller, LDO/LSO, paired replay.

if reference deployable but compute fail:
  kernelization remains blocker.

if compute pass but reference frontier fail:
  useful/null decision geometry remains blocker.

if useful high-AUC remains tiny-coverage:
  reset useful/control target decomposition again or redesign output-delta sufficient statistics.

if null remains inseparable:
  exact branch-delta remains diagnostic until null decomposition improves.

if oracle support collapses:
  carrier/support stability is blocker.
```
