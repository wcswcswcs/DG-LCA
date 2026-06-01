# DG-KAN v9.2.64 Conditional Null Separation 与 Support-Family Densification 完整实验计划

> 本计划基于 v9.2.63 `Conditional Bad-Event Separation 与 Risk-Adjusted Safe-Useful Frontier` 的真实复盘制定。  
> v9.2.63 的 terminal route 是：
>
> ```text
> route = R16-SupportRegression
> base_candidate = LQ-t2-h256
> success_v9263_strict_purekan_functional = False
> success_v9263_full_functional = False
> success_v9263_external_ready = False
> ```
>
> v9.2.63 的关键事实是：
>
> ```text
> P1:
>   conditional bad-event autopsy pass = 1
>   conditional bad count = 2309
>   risky useful count = 5358
>   harmless-null leak count = 9226
>   attribution fractions = 1.0
>
> P2:
>   best conditional bad statistic = CB5-FamilyInstabilityUCB
>   AUC = 0.810598
>   corr = 0.483450
>   conditional bad after gate = 0.0
>   coverage = 0.000496
>   utility pass = 0
>
> P3:
>   best safe-useful score = SU2-SafeUsefulMargin
>   AUC = 0.864844
>   coverage = 0.032201
>   bad-event = 0.029525
>   precision = 0.110398
>   null rate = 0.847240
>   utility pass = 0
>
> P4:
>   natural rows = 24192
>   balanced diagnostic rows = 6000
>   signal strata = 25
>   duplicate rows = 0
>   measured family count = 538 < 700
>   support confidence pass = 0
>   accepted support pass = 1
>   accepted strata = 9
>   accepted families = 79
>
> P5:
>   best exact reference frontier precision = 1.0
>   bad-event = 0.0
>   null rate = 0.0
>   coverage = 0.000124
>   precision LCB = 0.438494
>   bad-event UCB = 0.561506
>   accepted strata/families = 2/2
>   exact reference deployable = 0
>
> P6:
>   best = TBD0-V9256CBD0Reference
>   true-delta AUC = 0.879072
>   safe-useful AUC = 0.847784
>   conditional-bad AUC = 0.835101
>   agreement = 1.0
>   step ratio q90 = 2.863280 > 1.50
>   memory ratio = 0.969501
>   true-delta compute pass = 0
>
> Downstream:
>   P7-P10 not_run
>   reason = P5_reference_deployable_frontier_failed / P4_support_confidence_failed
> ```
>
> v9.2.64 的核心判断是：
>
> $$
> \boxed{
> \text{conditional bad-event 已经有强信号；当前最大 blocker 是 low-bad region 中 harmless-null 主导，以及 support-family confidence 回退。}
> }
> $$
>
> v9.2.63 不是“没有进展”。它证明了：
>
> $$
> \boxed{
> \text{bad-event 可以被 conditional risk gate 压住，但 strict gate 会变成 tiny slice。}
> }
> $$
>
> $$
> \boxed{
> \text{safe-useful score 可以同时达到 coverage 和 bad-event point estimate，但大部分 accepted rows 是 harmless-null。}
> }
> $$
>
> $$
> \boxed{
> \text{support accepted subset 本身平衡，但 measured family coverage 从 v9.2.62 回退。}
> }
> $$
>
> 因此 v9.2.64 不应继续只做：
>
> ```text
> 再加一个 conditional-bad score；
> 再硬化 bad-event gate；
> 再调 SU2 threshold；
> 再扩大 raw support rows；
> 再做 kernel-only；
> 按 dataset 单独调 controller。
> ```
>
> 本轮必须直接回答：
>
> $$
> \boxed{
> \text{在 low-bad / support-stable region 中，harmless-null 是否有 legal conditional signature？}
> }
> $$
>
> 如果 conditional null 可分，则构造 **null-aware safe-useful frontier**。  
> 如果 conditional null 不可分，则当前 exact branch-delta value signal 仍然只能作为 diagnostic signal，不能通过阈值小调变成 functional controller。

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

允许按 dataset 做诊断：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report conditional null and conditional bad rates by dataset
  report safe-useful / harmless-null / risky-useful distributions by dataset
  report support family density by dataset
  report leave-dataset-out failure matrix
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: use threshold A
  if dataset == KMNIST: use null rejector B
  if dataset == MNIST: skip high-risk rows
  tune threshold per dataset
  choose support family per dataset
  choose controller per dataset
  choose selected/full-logit mode per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-support-family-level / branch-delta features。Dataset name 只能进入 report，不能进入 commit rule。

---

# Part I. 对 v9.2.63 的独立判断

## 1. v9.2.63 没有达到目标

v9.2.63 没有 strict PureKAN functional success。失败发生在三道前置门：

```text
support_confidence_pass = 0
exact_reference_deployable = 0
true_delta_compute_pass = 0
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
paired replay success
```

P5 的 best reference frontier 是：

```text
precision = 1.0
bad-event = 0.0
null_rate = 0.0
coverage = 0.000124
```

这个结果不是 near-pass。它是 clean-too-tiny slice。它甚至比 v9.2.62 的 best reference coverage `0.016452` 更小，说明组合 gates 在 v9.2.63 中变得过度保守。

## 2. v9.2.63 的真实进展

v9.2.63 仍有三项重要进展。

第一，conditional bad-event autopsy 闭合。P1 的 conditional bad、risky useful、harmless-null 三类 attribution fraction 都是 `1.0`，说明 accepted-region failure 不再是黑箱。

第二，conditional bad-event statistic 有强信号。`CB5-FamilyInstabilityUCB` 达到：

$$
AUC_{\text{conditional-bad}}=0.810598,
$$

$$
Corr_{\text{conditional-bad}}=0.483450.
$$

这说明 useful-positive 区域里的 bad-event 并不是完全不可预测。

第三，safe-useful score 出现一个关键诊断结果。`SU2-SafeUsefulMargin` 达到：

$$
Coverage=0.032201,
$$

$$
BadEventRate=0.029525.
$$

这两个指标都满足 point-estimate gate，但：

$$
Precision=0.110398,
$$

$$
NullRate=0.847240.
$$

这说明当前已经不是 “找不到低 bad-event coverage” 的阶段，而是 “低 bad-event coverage 几乎全是 harmless-null” 的阶段。

## 3. v9.2.63 的真实失败

v9.2.63 的失败不是 conditional bad-event 不可预测，而是：

$$
\boxed{
\text{low-bad / coverage-pass region 被 harmless-null 主导。}
}
$$

P2 的 conditional bad gate 能把 bad-event 压到 `0.0`，但 coverage 只有 `0.000496`。  
P3 的 SU2 能保 coverage 与 bad-event，但 null rate 是 `0.847240`。  
P5 把 conditional bad、null、support 都严起来后，precision/bad/null 都干净，但 coverage 只有 `0.000124`。

这三个结果合起来说明：

```text
1. Bad-event separable，但 strict bad gate 太窄；
2. Broad low-bad region 存在，但 mostly-null；
3. Null/bad/support 三个 gates 同时严格时，region collapse 成 tiny slice。
```

因此，继续调一个单一 threshold 不会解决问题。我们需要的是 **conditional-null separation under low-bad support-stable region**。

## 4. 当前 blocker 的本质

当前 blocker 应从：

```text
support_stability_regression
```

进一步解释为：

$$
\boxed{
\text{conditional null separation failure + support-family confidence regression。}
}
$$

更具体地说：

```text
1. exact branch-delta signal 仍强；
2. conditional bad-event signal 也强；
3. support accepted subset 平衡；
4. broad safe-looking region 中 null 过高；
5. strict clean frontier 退化为 tiny slice；
6. measured family count 从 v9.2.62 的 pass 状态回退到 538；
7. true-delta compute 仍太贵。
```

因此 v9.2.64 必须同时做两件事：

```text
A. 在 low-bad / support-stable region 中，分解 harmless-null submodes；
B. 恢复 support-family measured coverage，并用 family confidence 做 LCB/UCB 而不是盲目扩 support。
```

## 5. 是否还在正确道路上

是，但需要再次收紧路线。

正确路线现在是：

```text
support accepted subset pass
→ conditional bad-event signal pass
→ conditional null separation
→ support-family densification
→ null-aware safe-useful frontier
→ exact reference deployable frontier
→ true-delta system path
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线是：

```text
继续只扩 support rows；
继续硬化 conditional bad gate；
继续只调 SU2 threshold；
继续把 SU2 coverage/bad-event point estimate 写成 success；
继续把 P5 clean tiny slice 写成 success；
继续只做 true-delta kernel；
按 MNIST/Fashion/KMNIST 分别调 gate。
```

---

# Part II. v9.2.64 总体目标

v9.2.64 的总体目标是：

$$
\boxed{
\text{在 low-bad / support-stable region 中分离 harmless-null，并恢复 support-family confidence，构造可部署 safe-useful frontier。}
}
$$

本轮必须并行验证六件事：

```text
1. 复现 v9.2.63 boundary。
2. 对 SU2 / P5 accepted rows 做 conditional-null autopsy。
3. 建立 legal conditional-null rejector。
4. 恢复 measured family count 与 support confidence。
5. 在 CBD0 exact reference 上构造 null-aware safe-useful frontier。
6. 并行继续 true-delta compute v9，但不让 compute lane 掩盖 decision blocker。
```

本轮最关键 stop-go 不是 “conditional bad AUC 是否更高”，而是：

$$
\boxed{
\text{能否把 SU2 的 null rate 从 }0.847240\text{ 降到 }\leq0.15\text{，同时保持 coverage}\geq0.03\text{ 与 bad-event}\leq0.05？
}
$$

如果不能，说明当前 exact branch-delta / safe-useful representation 缺少足够的 null/useful separation。此时继续 kernelization 仍没有 functional success 意义。

---

# Part III. 核心假设

## H1：v9.2.63 的 P3 failure 是 conditional-null failure，而不是 bad-event failure

证据：

```text
SU2 coverage = 0.032201
SU2 bad-event = 0.029525
SU2 precision = 0.110398
SU2 null rate = 0.847240
```

H1 成立标准：

在 SU2-like low-bad region 内加入 conditional-null rejector 后：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\geq0.03,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
NullRate_{\text{heldout}}\leq0.15.
$$

H1 失败标准：

conditional-null rejector 无法把 null rate 降到 `0.15` 以下，或一降低 null rate，coverage 回落到 `<0.02`。

## H2：conditional null 是多模式混合，不应使用一个 global null score

v9.2.63 的 P3 null dominance 可能来自多种模式：

```text
N1-control-equivalent:
  useful score 看似高，但 RealFunctional 没有强于 AdamWParallel / bestLR。

N2-low-real-gain:
  RealFunctional 自身 gain 不足。

N3-low-persistence:
  h20/h80 有短期信号，但 h240/h640 消失。

N4-delta-silent:
  logit displacement / delta norm 不足。

N5-support-only:
  support/family 很稳定，但 value 没有出现。

N6-risk-suppressed-null:
  conditional bad 很低，但 value 也低。

N7-family-average-null:
  family reliability 高，但 individual event useful value 低。

N8-score-artifact-null:
  SU2 margin high 来自 score construction，而不是 grounded useful event。
```

H2 成立标准：

至少四个 null submodes 有 legal statistic 达到：

$$
AUC_{\text{null-submode}}\geq0.65
$$

or:

$$
Corr_{\text{null-submode}}\geq0.30.
$$

组合 rejector 达到：

$$
NullRate_{\text{accepted}}\leq0.15,
$$

$$
Coverage\geq0.03.
$$

H2 失败标准：

所有 null submode statistic 都低于 diagnostic threshold，或组合 rejector 只产生 tiny clean slice。

## H3：P4 support regression 来自 family definition / bucket sparsity，而不是 raw row count 不足

v9.2.63 的 support：

```text
natural rows = 24192
balanced rows = 6000
signal strata = 25
duplicate rows = 0
measured family count = 538 < 700
accepted support pass = 1
```

H3 成立标准：

通过 family densification / bucket rebalance 达到：

```text
natural_real_event_count >= 24192
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 25
measured_family_count >= 700
duplicate_row_count = 0
```

并且 accepted set 仍满足：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

H3 失败标准：

raw rows 足够但 family count 仍低于 `700`，或 family count 提升后 accepted support balance 崩溃。

## H4：P5 clean-too-tiny 是 over-constrained composition，不是 absence of safe-useful support

v9.2.63 P5：

```text
precision = 1.0
bad-event = 0.0
null_rate = 0.0
coverage = 0.000124
accepted strata/families = 2/2
```

H4 认为 safe-useful support 不是没有，而是当前 gates 组合过度保守。

H4 成立标准：

null-aware frontier 达到：

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H4 失败标准：

所有 frontier 仍只有：

```text
coverage < 0.01
```

或一扩 coverage 就 precision / bad-event 崩溃。

## H5：true-delta compute 仍是并行 blocker，但不是 v9.2.64 的主 blocker

v9.2.63 P6：

```text
true_delta_auc = 0.879072
safe_useful_auc = 0.847784
conditional_bad_auc = 0.835101
agreement = 1.0
step ratio q90 = 2.863280
memory ratio = 0.969501
```

H5 成立标准：

至少一个 true-delta compute candidate 达到：

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

H5 失败标准：

true-delta compute 仍太贵，或者 signal lost。  
但如果 H1-H4 失败，即使 H5 成立，也不能进入 functional success。

---

# Part IV. 新目标分解

## 1. 从四类到 conditional-null

v9.2.63 已经强调四类：

```text
safe-useful
risky-useful
harmless-null
bad-null
```

v9.2.64 进一步引入：

$$
Y_{\text{null}\mid L}
=
Y_{\text{null}}
\quad
\text{conditioned on}
\quad
L(e)=LowBad(e)\land SupportStable(e).
$$

其中 $L(e)$ 是 low-bad / support-stable candidate region，例如：

$$
L(e)=
\mathbb{1}[UCB_{\text{bad-cond}}(e)\leq\rho_B]
\land
\mathbb{1}[Rel_{\text{support}}(family(e))\geq r_0].
$$

## 2. Null-aware safe-useful score

旧 v9.2.63 score：

$$
S_{\text{safe-useful}}
=
LCB_{\text{value-control}}
-
\lambda_B UCB_{\text{bad-cond}}
-
\lambda_N P_{\text{null}}
+
\lambda_S Rel_{\text{support}}.
$$

v9.2.64 改为 conditional form：

$$
S_{\text{safe-useful-null-aware}}
=
LCB_{\text{value-control}}
-
\lambda_B UCB_{\text{bad}\mid U,\neg N,S}
-
\lambda_N UCB_{\text{null}\mid L}
+
\lambda_S LCB_{\text{support}}
+
\lambda_D Density_{\text{useful-pocket}}.
$$

Accept：

$$
Accept(e)
=
\mathbb{1}[S_{\text{safe-useful-null-aware}}(e)\geq\tau]
\land
\mathbb{1}[UCB_{\text{null}\mid L}(e)\leq\rho_N]
\land
\mathbb{1}[UCB_{\text{bad}\mid U,\neg N,S}(e)\leq\rho_B]
\land
SupportStable(e).
$$

Calibration objective：

$$
\max Coverage(A)
$$

subject to：

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
accepted\_signal\_strata\_count(A)\geq5,
$$

$$
accepted\_family\_count(A)\geq32,
$$

$$
max\_family\_share(A)\leq0.50.
$$

---

# Part V. Candidate statistics

## 1. Conditional-null statistics

### CN0：v9.2.63 null reference

Reference only.

### CN1：ControlEquivalentNullUCB

$$
S_{\text{control-null}}
=
UCB(Gain_{\text{controls}})
-
LCB(Gain_F).
$$

High score means functional gain is not control-resistant.

### CN2：LowRealGainNullLCB

$$
S_{\text{lowgain-null}}
=
-\operatorname{LCB}(Gain_F).
$$

High score means RealFunctional gain is weak.

### CN3：PersistenceNullRisk

$$
S_{\text{persistence-null}}
=
\max(0, Gap_{20})
+
\max(0, Gap_{80})
-
\min(Gap_{240},Gap_{640}).
$$

High score means short-horizon apparent value but weak persistent value.

### CN4：DeltaSilentNull

$$
S_{\text{delta-silent}}
=
-\frac{\|\Delta z_F\|}
{\operatorname{median}(\|\Delta z_F\|)+\epsilon}.
$$

High score means event is too silent in output space.

### CN5：SupportOnlyNull

$$
S_{\text{support-only-null}}
=
LCB_{\text{support}}
-
LCB_{\text{value-control}}.
$$

High score means support is stable but value is absent.

### CN6：FamilyAverageNullUCB

$$
UCB_{\text{null}}(f)
=
\hat{p}_{null}(f)
+
\kappa
\sqrt{
\frac{\hat{p}_{null}(f)(1-\hat{p}_{null}(f))}{n_f+\epsilon}
}.
$$

### CN7：ConditionalNullMixture

Monotone mixture:

$$
UCB_{\text{null}\mid L}
=
c_1S_{\text{control-null}}
+
c_2S_{\text{lowgain-null}}
+
c_3S_{\text{persistence-null}}
+
c_4S_{\text{delta-silent}}
+
c_5S_{\text{support-only-null}}
+
c_6UCB_{\text{null}}(f).
$$

Coefficients selected on calibration split only, no dataset name.

---

## 2. Safe-useful statistics

### SU0：v9.2.63 SU2 reference

Reference only.

### SU1：NullAdjustedSafeUsefulMargin

$$
S_{\text{NA-SU}}
=
S_{\text{SU2}}
-
\lambda_N UCB_{\text{null}\mid L}
-
\lambda_B UCB_{\text{bad-cond}}.
$$

### SU2：ValueControlLCBPlusNullPenalty

$$
S_{\text{VCN}}
=
LCB(Gain_F)
+
LCB(Gain_F-\max Gain_{\text{controls}})
-
\lambda_N UCB_{\text{null}\mid L}.
$$

### SU3：PersistentNoRegretNullAdjusted

$$
S_{\text{PNR}}
=
\min_{h\in\mathcal{H}}
[
Gain_{F,h}
-
\max_{c\in\mathcal{C}}Gain_{c,h}
]
-
\lambda_N UCB_{\text{null}\mid L}
-
\lambda_B UCB_{\text{bad-cond}}.
$$

Controls:

```text
AdamWParallel
bestLR
NoOp
RandomMatched
LRScaled
TrustRatioScaled
```

### SU4：UsefulPocketDensityScore

Feature vector:

```text
value-control LCB
conditional bad UCB
conditional null UCB
support reliability
delta efficiency
horizon persistence
control gap
role bucket
stratum bucket
```

Density score:

$$
Density_{\text{useful-pocket}}(e)
=
\frac{k}
{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### SU5：HybridNullAwareSafeUseful

$$
S_{\text{hybrid}}
=
a_1S_{\text{NA-SU}}
+
a_2S_{\text{VCN}}
+
a_3S_{\text{PNR}}
+
a_4Density_{\text{useful-pocket}}
+
a_5LCB_{\text{support}}.
$$

---

## 3. Support-family densification candidates

### SF0：v9.2.63 support reference

Reference only.

### SF1：FamilyBucketRebalance

Family definition:

$$
family(e)=
(
stratum,
horizon,
role,
carrier,
value\_bucket,
null\_bucket,
bad\_bucket,
support\_bucket,
delta\_bucket
).
$$

Bucket boundaries are fixed on calibration split and then frozen.

### SF2：HierarchicalFamilyBackoff

Coarse / mid / fine families:

```text
coarse:
  stratum, horizon, role

mid:
  stratum, horizon, role, value_bucket, bad_bucket

fine:
  stratum, horizon, role, value_bucket, bad_bucket, null_bucket, delta_bucket
```

Empirical-Bayes reliability:

$$
Rel_{\text{EB}}(f)
=
\frac{n_f}{n_f+\alpha}\hat{p}_f
+
\frac{\alpha}{n_f+\alpha}Rel(parent(f)).
$$

### SF3：NullAwareFamilyReliability

$$
Rel_{\text{safe-useful}}(f)
=
LCB[
P(Y_{\text{safe-useful}}=1\mid f)
]
-
\lambda_N UCB[
P(Y_{\text{null}}=1\mid f)
]
-
\lambda_B UCB[
P(Y_{\text{bad}}=1\mid f)
].
$$

### SF4：SupportDensificationSamplerAudit

This is not a training sampler. It is a measurement generator audit. It ensures balanced diagnostic rows cover under-filled family buckets by collecting real train-stream event rows.

Required:

```text
duplicate_row_count = 0
source_hash present
no synthetic row
no dataset-name branch
```

### SF5：LeaveFamilyStratumOutConfidence

$$
Rel_{\text{LFSO}}(f)
=
\min(
Rel(f),
\min_{f'\in\mathcal{N}(f)}Rel(f'),
\min_{s'\in\mathcal{N}(s)}Rel(s')
).
$$

---

# Part VI. Controller candidates

## C0：v9.2.63 best reference

Reference only.

## C1：SU2PlusConditionalNullRejector

Start from SU2-like low-bad region:

$$
Accept(e)
=
SU2(e)
\land
UCB_{\text{null}\mid L}(e)\leq\rho_N
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
SupportStable(e).
$$

## C2：NullAdjustedSafeUsefulController

$$
Accept(e)
=
S_{\text{NA-SU}}(e)\geq\tau
\land
UCB_{\text{null}\mid L}(e)\leq\rho_N
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
Rel_{\text{safe-useful}}(family(e))\geq r_0.
$$

## C3：FourClassNullAwareController

Four-class triage:

$$
Accept(e)
=
\mathbb{1}
[
P_{\text{safe-useful}}(e)
-
\max(
P_{\text{risky-useful}}(e),
P_{\text{harmless-null}}(e),
P_{\text{bad-null}}(e)
)
\geq\tau_4
].
$$

## C4：UsefulPocketController

$$
Accept(e)
=
Density_{\text{useful-pocket}}(e)\geq d_0
\land
S_{\text{hybrid}}(e)\geq\tau
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
UCB_{\text{null}\mid L}(e)\leq\rho_N.
$$

## C5：ConstrainedNullAwareOptimizer

Search thresholds to maximize:

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
NullRate(A)\leq0.15,
$$

$$
Coverage(A)\in[0.03,0.15],
$$

$$
accepted\_signal\_strata\_count(A)\geq5,
$$

$$
accepted\_family\_count(A)\geq32.
$$

## C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.63 boundary reproduction

### 目标

确认 v9.2.63 boundary 稳定，尤其是 P3 null dominance、P4 support regression、P5 clean-too-tiny。

### 必须记录

```text
route
source_route_v9262
conditional_bad_autopsy_pass
conditional_bad_stat_pass
safe_useful_score_pass
support_confidence_pass
accepted_support_pass
exact_reference_deployable
true_delta_compute_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R16-SupportRegression
P3 SU2 null dominance reproduced
P4 measured family count < 700 reproduced or explicitly changed
P5 clean-too-tiny reproduced
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_v9262_to_v9263_progress_ladder.svg
p0_null_bad_support_failure_ladder.svg
```

---

## P1：conditional-null autopsy inside low-bad regions

### 目标

解释 SU2 为什么 coverage/bad-event 过线但 precision 极低。重点是拆 harmless-null，而不是继续泛泛分析 bad-event。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
accepted_by
low_bad_region
support_stable
safe_useful
risky_useful
harmless_null
bad_null
bad_event
null_event
control_equivalent_null
low_real_gain_null
nonpersistent_null
delta_silent_null
support_only_null
family_average_null
score_artifact_null
gap_score
safe_useful_score
conditional_bad_score
conditional_null_score
support_score
```

Conditional rates 必须报告：

```text
P(null | SU2)
P(bad | SU2)
P(safe_useful | SU2)

P(null | SU2, conditional_bad_safe)
P(bad | SU2, conditional_bad_safe)
P(safe_useful | SU2, conditional_bad_safe)

P(null | low_bad, support_stable)
P(safe_useful | low_bad, support_stable)
```

### 判断标准

P1 pass：

```text
conditional_null_attribution_fraction >= 0.90
safe_useful_miss_attribution_fraction >= 0.90
bad_event_attribution_fraction >= 0.90
at least four null submodes have nonzero support
```

### 可视化

```text
p1_low_bad_region_sankey.svg
p1_null_submode_breakdown.svg
p1_su2_accepted_decomposition.svg
p1_safe_useful_vs_null_score_scatter.svg
p1_conditional_rates_table.svg
```

---

## P2：conditional-null statistic factory

### 目标

建立 legal online conditional-null rejector，专门分离 safe-useful 与 harmless-null。

### 必须记录

```text
null_stat_id
conditional_region
null_submode
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_conditional_null
corr_conditional_null
AUC_submode
null_rate_after_gate
coverage_after_gate
precision_after_gate
bad_event_after_gate
calibration_error
feature_overhead
memory_overhead
```

### 判断标准

Conditional-null statistic pass：

$$
AUC_{\text{null}\mid L}\geq0.70
$$

or:

$$
Corr_{\text{null}\mid L}\geq0.35.
$$

Submode diagnostic pass：

At least four submodes satisfy:

$$
AUC_{\text{submode}}\geq0.65
$$

or:

$$
Corr_{\text{submode}}\geq0.30.
$$

Utility pass：

$$
NullRate\leq0.15,
$$

$$
Coverage\geq0.03,
$$

$$
Precision\geq0.75,
$$

$$
BadEventRate\leq0.05.
$$

Diagnostic pass：

$$
NullRate\leq0.30
$$

with:

$$
Coverage\geq0.02
$$

and:

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p2_conditional_null_auc_matrix.svg
p2_null_rejection_frontier.svg
p2_null_submode_roc.svg
p2_safe_useful_null_bad_triage.svg
p2_null_calibration_curve.svg
```

---

## P3：support-family densification and confidence recovery

### 目标

恢复 P4 support confidence family count，同时保持 accepted support balance。此阶段不能简单复制 rows，也不能按 dataset 造 family。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata target >= 25
family target >= 700
row_sources = natural, balanced_diagnostic
```

Balanced diagnostic rows 必须是真实 train-stream events，不能复制自然 rows。

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family_coarse
event_family_mid
event_family_fine
source_hash
duplicate_row_flag
safe_useful
harmless_null
bad_event
conditional_bad_score
conditional_null_score
support_density
family_reliability
family_bucket_fill_status
```

### 判断标准

Support confidence pass：

```text
natural_real_event_count >= 24192
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 25
measured_family_count >= 700
duplicate_row_count = 0
```

Accepted support pass：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p3_family_densification_heatmap.svg
p3_family_bucket_fill_matrix.svg
p3_signal_strata_coverage.svg
p3_accepted_family_balance.svg
p3_hierarchical_family_backoff.svg
```

---

## P4：null-aware safe-useful score factory

### 目标

在 low-bad / support-stable region 中，组合 conditional-null、conditional-bad、value-control 和 support confidence，构造 coverage-preserving safe-useful score。

### 必须记录

```text
safe_useful_score_id
conditional_null_stat_id
conditional_bad_stat_id
value_stat_id
support_stat_id
coefficients
thresholds
AUC_safe_useful
corr_safe_useful
precision_after_score
coverage_after_score
bad_event_after_score
null_rate_after_score
precision_lcb
bad_event_ucb
accepted_strata_count
accepted_family_count
```

### 判断标准

Score pass：

$$
AUC_{\text{safe-useful}}\geq0.70
$$

or:

$$
Corr_{\text{safe-useful}}\geq0.35.
$$

Utility pass：

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

### 可视化

```text
p4_null_aware_safe_useful_frontier.svg
p4_value_null_bad_surface.svg
p4_lcb_ucb_constraint_curve.svg
p4_score_ablation.svg
```

---

## P5：exact reference deployable frontier v8

### 目标

在 CBD0 exact reference 上判断 null-aware safe-useful frontier 是否可部署。P5 是本轮最重要 gate。

### 必须记录

```text
controller_id
safe_useful_score_id
conditional_null_stat_id
conditional_bad_stat_id
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
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

If no controller passes, route must be one of:

```text
R10-ConditionalNullInseparable
R11-ReferenceStillTinyAfterNullRepair
R12-SupportConfidenceStillRegressed
```

### 可视化

```text
p5_reference_frontier_v8_precision_coverage_bad_null.svg
p5_null_aware_threshold_surface.svg
p5_deployable_region_ladder.svg
p5_oracle_legal_gap_after_null_repair.svg
```

---

## P6：true-delta compute v9 parallel lane

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
AUC_safe_useful
AUC_conditional_bad
AUC_conditional_null
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
safe_useful_component_time
conditional_bad_component_time
conditional_null_component_time
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
p6_true_delta_v9_cost_signal_pareto.svg
p6_compute_vs_decision_ladder.svg
p6_residual_subphase_after_conditional_null.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P5 reference deployable pass 与 P6 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
safe_useful_score_id
conditional_null_stat_id
conditional_bad_stat_id
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
safe_useful_score_id
conditional_null_stat_id
conditional_bad_stat_id
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
safe_useful_score_id = best P8 survivor
conditional_null_stat_id = best P8 survivor
conditional_bad_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledSafeUsefulScore, ShuffledConditionalNull,
           ShuffledConditionalBad, ShuffledSupportStat, ShuffledControlGain,
           ShuffledCandidateGate, ShuffledBranchRatio, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
safe_useful_score_id
conditional_null_stat_id
conditional_bad_stat_id
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
ShuffledSafeUsefulScore = fail
ShuffledConditionalNull = fail
ShuffledConditionalBad = fail
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
contract_audit_v9264.csv
p0_v9263_boundary_reproduction.csv
p1_conditional_null_autopsy.csv
p2_conditional_null_stat_factory.csv
p3_support_family_densification_confidence_recovery.csv
p4_null_aware_safe_useful_score_factory.csv
p5_exact_reference_deployable_frontier_v8.csv
p6_true_delta_compute_v9_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
conditional_null_trace_v9264.csv
null_submode_trace_v9264.csv
support_family_densification_trace_v9264.csv
null_aware_safe_useful_score_trace_v9264.csv
reference_frontier_v8_trace.csv
true_delta_compute_v9_trace.csv
system_controller_trace_v9264.csv
leaveout_trace_v9264.csv
paired_replay_branch_trace_v9264.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9263_boundary_unstable
F3_dataset_tuning_detected
F4_conditional_null_autopsy_incomplete
F5_conditional_null_inseparable
F6_null_submode_unattributed
F7_support_family_densification_fail
F8_support_confidence_regression
F9_null_aware_score_not_predictive
F10_null_aware_score_tiny_coverage
F11_bad_event_ucb_fail
F12_precision_lcb_fail
F13_null_rate_fail
F14_exact_reference_still_not_deployable_after_null_repair
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
  v9.2.63 boundary is reproduced.

R2-ConditionalNullAutopsyPass:
  harmless-null modes inside low-bad region are attributed.

R3-ConditionalNullStatisticPass:
  conditional-null predictor separates safe-useful from harmless-null.

R4-SupportFamilyDensificationPass:
  measured family count and support confidence recover.

R5-NullAwareSafeUsefulScorePass:
  safe-useful score is predictive and coverage-preserving.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + conditional-null / conditional-bad / support frontier passes precision / coverage / bad-event / null gate.

R7-TrueDeltaComputeV9SystemPass:
  true branch-delta v9 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-ConditionalNullInseparable:
  low-bad/support-stable region contains harmless-null without legal predictable signature.

R13-SafeUsefulStillTiny:
  null-aware score can be safe only at coverage < 0.03.

R14-SupportConfidenceStillRegressed:
  family count / support confidence remains below gate.

R15-ReferenceFeasibleButComputeFail:
  decision geometry viable, true-delta compute still too expensive.

R16-ComputePassButReferenceFail:
  compute viable, accept-region geometry still fails.

R17-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R18-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R19-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R20-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9263_boundary_pass
dataset_tuning_detected
conditional_null_autopsy_pass
conditional_null_attribution_fraction
safe_useful_miss_attribution_fraction
bad_event_attribution_fraction
best_conditional_null_stat_id
conditional_null_stat_pass
conditional_null_auc
conditional_null_corr
null_rate_after_gate
coverage_after_null_gate
best_support_family_stat_id
support_family_densification_pass
support_confidence_pass
natural_real_event_count
balanced_diagnostic_real_event_count
measured_signal_strata_count
measured_family_count
duplicate_row_count
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
best_safe_useful_score_id
null_aware_safe_useful_score_pass
safe_useful_auc
safe_useful_precision
safe_useful_coverage
safe_useful_bad_event
safe_useful_null_rate
precision_lcb
bad_event_ucb
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
true_delta_safe_useful_auc
true_delta_conditional_bad_auc
true_delta_conditional_null_auc
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
success_v9264_strict_purekan_functional
success_v9264_full_functional
success_v9264_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 conditional-null autopsy
  P2 conditional-null statistic factory
  P3 support-family densification
  P4 null-aware safe-useful score factory
  P5 exact reference deployable frontier v8
  P6 true-delta compute v9

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
v9.2.63 boundary reproduced
conditional-null autopsy completed
conditional-null statistic measured
support-family densification measured
null-aware safe-useful score measured
exact reference deployable frontier v8 measured
true-delta compute v9 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
conditional-null statistic pass
+
support confidence pass
+
exact reference deployable frontier pass
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
1. v9.2.63 boundary cannot be reproduced；
2. conditional-null autopsy cannot attribute harmless-null rows；
3. conditional-null has no legal predictable signature；
4. null-aware score remains tiny coverage；
5. null rate remains above 0.15 at coverage >=0.03；
6. coverage crosses 0.03 but bad-event UCB remains above 0.05；
7. precision LCB remains below 0.75；
8. support confidence remains below measured family gate；
9. exact reference remains not deployable after conditional-null repair；
10. true branch-delta compute remains too expensive；
11. true branch-delta compute passes system but loses signal；
12. system-legal controller cannot meet precision / coverage / bad-event / null-rate；
13. leave-dataset-out fails；
14. leave-stratum-out fails；
15. paired replay remains control-equivalent；
16. shuffle controls pass, indicating overfit；
17. short-run task drops；
18. full run gives no macro / hard-stratum / geometry gain；
19. functional breaks system gate；
20. gains are explained by QuadraticFeatureMLP；
21. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：conditional-null + support confidence + reference frontier + compute + LDO/LSO + paired replay pass

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

下一步继续 kernel / layout / fused null-aware safe-useful compute，不调 dataset。

## Case C：compute pass but reference frontier fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 conditional-null / support-family sufficient statistics，不走 kernel-only。

## Case D：conditional-null inseparable

必须声明：

```text
low-bad support-stable region contains harmless-null that current legal features cannot separate.
```

下一步重设 value/useful target or output-delta sufficient statistics。

## Case E：support confidence still regresses

必须声明：

```text
support accepted subset is balanced, but family confidence is insufficient for official promotion.
```

下一步修 family definition / bucket densification / generator coverage，不调 dataset。

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

v9.2.64 的一句话策略是：

$$
\boxed{
\text{不要再只压 bad-event；现在要在 low-bad/support-stable 区域内分离 harmless-null，并恢复 family confidence。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
conditional bad AUC 是否存在；
support accepted subset 是否平衡；
SU2 是否有 coverage/bad point estimate；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. SU2 accepted rows 的 harmless-null 到底是哪几类？
2. 在 low-bad / support-stable 条件下，null 是否可预测？
3. control-equivalent / low-real-gain / nonpersistent / delta-silent / support-only 哪个是主因？
4. conditional-null UCB 能否把 null rate 从 0.847240 降到 <=0.15？
5. coverage 能否保持 >=0.03？
6. measured family count 能否从 538 恢复到 >=700？
7. exact-reference frontier 能否从 coverage 0.000124 推到 >=0.03？
8. precision LCB 能否从 0.438494 推到 >=0.75？
9. bad-event UCB 能否从 0.561506 压到 <=0.05？
10. true delta compute v9 能否继续压到 step <=1.50？
11. system-legal controller 能否 LDO/LSO？
12. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.64 的结果将给出清晰分叉：

```text
if conditional-null separates and reference frontier becomes deployable:
  decision geometry is viable; proceed to true-delta compute and official controller.

if conditional-null separates but support confidence fails:
  support-family densification remains blocker.

if reference deployable but compute fails:
  kernelization remains blocker.

if conditional-null cannot separate:
  current exact branch-delta signal remains diagnostic; redesign value/useful/output-delta sufficient statistics.

if both reference and compute pass:
  open LDO/LSO and official paired replay.
```
