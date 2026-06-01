# DG-KAN v9.2.63 Conditional Bad-Event Separation 与 Risk-Adjusted Safe-Useful Frontier 完整实验计划

> 本计划基于 v9.2.62 `Useful-Control Target Reset 与 Null-Rejection Frontier` 的真实复盘制定。  
> v9.2.62 的 terminal route 是：
>
> ```text
> route = R12-UsefulTargetStillTiny
> base_candidate = LQ-t2-h256
> success_v9262_strict_purekan_functional = False
> success_v9262_full_functional = False
> success_v9262_external_ready = False
> ```
>
> v9.2.62 的关键事实是：
>
> ```text
> P0:
>   v9.2.61 boundary reproduced
>   exact reference deployable = 0
>   true-delta compute pass = 0
>   fake/proxy/offload = 0
>
> P1:
>   useful-control target autopsy pass = 1
>   accepted count = 5997
>   useful rejected = 230
>   harmless-null accepted = 684
>   bad-event accepted = 2846
>   attribution fractions = 1.0
>
> P2:
>   predictive null submode count = 4
>   best null rejector = N2-LowGainNullRejector
>   null AUC = 0.720924
>   null rate after gate = 0.139792
>   diagnostic pass = 1
>   official pass = 0
>
> P3:
>   best useful/control statistic = U1-RealGainLCBv2
>   useful AUC = 0.703150
>   coverage = 0.062707
>   precision = 0.446935
>   bad-event = 0.466051
>   utility pass = 0
>
> P4:
>   support stability pass = 1
>   natural rows = 24192
>   balanced diagnostic rows = 6000
>   signal strata = 27
>   families = 741
>   duplicate rows = 0
>   accepted support pass = 1
>
> P5:
>   best exact reference controller =
>     C-U6-UsefulMixtureScore
>     + N2-LowGainNullRejector
>     + R2-UsefulConditionedBadRisk
>     + S4-CoverageBalancedSupportCap
>
>   exact_reference_deployable = 0
>   precision = 0.763819
>   coverage = 0.016452
>   bad-event = 0.236181
>   null_rate = 0.020101
>   precision_lcb = 0.719692
>   bad_event_ucb = 0.280308
>   accepted signal strata = 5
>   accepted family count = 34
>
> P6:
>   best = TBD0-V9256CBD0Reference
>   AUC = 0.879072
>   agreement = 1.0
>   step ratio q90 = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> Downstream:
>   P7-P10 = not_run
>   reason = P5/P3 gate-blocked
>
> current blocker:
>   useful_stat_precision_fail
> ```
>
> v9.2.63 的核心判断是：
>
> $$
> \boxed{
> \text{null 已经能压住，support 已经过关；当前最大问题是 useful-positive 区域里混入大量 conditional bad-events。}
> }
> $$
>
> v9.2.62 不是失败在 “没有信号”。它证明了：
>
> $$
> \boxed{
> \text{useful/control 有信号，但当前 useful gate 选到的是 useful-looking + high-risk rows。}
> }
> $$
>
> $$
> \boxed{
> \text{null rejection 有效，但 null rejection 不能替代 bad-event separation。}
> }
> $$
>
> $$
> \boxed{
> \text{support stability 已经不是第一 blocker，不能再把失败主要归因于 support 太窄。}
> }
> $$
>
> 因此 v9.2.63 不应继续只做：
>
> ```text
> 再扩 support rows；
> 再硬化 null rejector；
> 再调 useful score threshold；
> 再堆一个 hybrid score；
> 再做 kernel-only；
> 按 dataset 单独调 controller。
> ```
>
> 本轮必须直接回答：
>
> $$
> \boxed{
> \text{在 useful-positive / null-rejected / support-stable 条件下，bad-event 是否有 legal conditional signature？}
> }
> $$
>
> 如果 conditional bad-event 可分，则构造 **risk-adjusted safe-useful frontier**。  
> 如果 conditional bad-event 不可分，则当前 exact branch-delta signal 仍只是 diagnostic，不能靠阈值或 kernelization 变成 functional success。

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
  report conditional bad-event by dataset
  report useful-positive bad-event modes by dataset
  report null mode distribution by dataset
  report support/family reliability by dataset
  report leave-dataset-out failure matrix
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: use threshold A
  if dataset == KMNIST: use conditional-risk model B
  if dataset == MNIST: skip useful-positive rows
  tune thresholds per dataset
  choose support family per dataset
  choose controller per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-support-family-level / branch-delta features。Dataset name 只能进入 report，不能进入 commit rule。

---

# Part I. 对 v9.2.62 的独立判断

## 1. v9.2.62 没有达到目标

v9.2.62 没有 strict PureKAN functional success。它失败在 exact-reference deployable frontier 和 true-delta system gate：

```text
exact_reference_deployable = 0
true_delta_compute_pass = 0
system_legal_controller = not_run
LDO / LSO / paired replay = not_run
```

P5 best exact-reference controller 的 point estimate 看似有一项过线：

```text
precision = 0.763819
null_rate = 0.020101
accepted family count = 34
accepted signal strata = 5
```

但它不能 official，因为：

```text
coverage = 0.016452 < 0.03
bad-event = 0.236181 > 0.05
precision_lcb = 0.719692 < 0.75
bad_event_ucb = 0.280308 > 0.05
```

这不是 near-pass。bad-event 离 gate 差得很远，说明当前 accepted region 不是一个 “稍微扩大即可成功” 的区域。

## 2. v9.2.62 的真实进展

v9.2.62 有三项实质进展。

第一，support stability 已经过关，并且比 v9.2.61 更宽：

```text
natural rows = 24192
balanced diagnostic rows = 6000
signal strata = 27
families = 741
accepted signal strata = 11
accepted families = 71
duplicate rows = 0
```

这说明 support generator 不再是第一 blocker。后续如果 controller 失败，不能再简单说 “样本太少 / strata 太窄”。

第二，null rejection 有真实诊断信号。`N2-LowGainNullRejector` 的 null AUC 为 `0.720924`，并且能把 null rate 降到 `0.139792`。P5 最佳 frontier 甚至把 null rate 降到 `0.020101`。这说明 harmless-null 不再是唯一主要污染源。

第三，useful/control statistic 有 broad coverage 能力。`U1-RealGainLCBv2` 取得 `coverage = 0.062707`，已经超过 official coverage 下限 `0.03`。这说明 useful gate 可以放宽到足够 coverage，但放宽后 precision 只有 `0.446935`，bad-event 达到 `0.466051`。

## 3. v9.2.62 的真实失败

v9.2.62 的失败点非常清楚：

$$
\boxed{
\text{当前 useful-positive region 不是 safe-useful region，而是 high-risk useful-looking region。}
}
$$

P3 的 `U1-RealGainLCBv2`：

```text
coverage = 0.062707
precision = 0.446935
bad-event = 0.466051
```

这说明它会选到足够多的 rows，但这些 rows 中接近一半是 bad-event。  
P5 加入 null rejector、conditioned risk、support cap 后：

```text
null_rate = 0.020101
precision = 0.763819
bad-event = 0.236181
coverage = 0.016452
```

这说明 null 已经被压低，但 bad-event 仍然很高。因此继续强化 null rejector 不会解决主问题。

## 4. 当前问题的本质

当前 blocker 应从：

```text
useful_stat_precision_fail
```

进一步解释为：

$$
\boxed{
\text{conditional bad-event separation failure inside useful-positive / null-rejected / support-stable region。}
}
$$

现在应该把 accepted rows 拆成至少四类：

```text
A. safe-useful:
   有 functional value，强于 controls，且不会产生 bad-event。

B. risky-useful:
   useful score 高，但 tail harm / generalization harm / control instability 高。

C. harmless-null:
   安全但没有 useful functional value。

D. bad-null / bad-low-value:
   既没有 useful value，又有 bad-event risk。
```

v9.2.62 的 P5 已经把 C 类大幅压低；真正难的是把 A 与 B 分开。

因此下一步必须建模：

$$
P(Y_{\text{bad}}=1 \mid UsefulPositive, NullRejected, SupportStable).
$$

如果这个 conditional risk 不可预测，那么 exact branch-delta value signal 不能被当前 legal features 安全部署。

## 5. 是否还在正确道路上

是，但路线必须再次收紧。

正确路线现在是：

```text
support stability pass
→ null rejection diagnostic pass
→ useful broad coverage observed
→ conditional bad-event separation
→ risk-adjusted safe-useful frontier
→ exact reference deployable frontier
→ true-delta system path
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线是：

```text
继续扩 support rows；
继续硬化 null rejector；
继续只调 useful threshold；
继续把 U1 coverage 当成功；
继续把 P5 precision point estimate 当成功；
继续只做 true-delta kernel；
按 dataset 分别调 gate。
```

---

# Part II. v9.2.63 总体目标

v9.2.63 的总体目标是：

$$
\boxed{
\text{在 useful-positive / null-rejected / support-stable 条件下分离 conditional bad-events，并构造 risk-adjusted safe-useful frontier。}
}
$$

本轮要同时完成六件事：

```text
1. 复现 v9.2.62 boundary。
2. 对 P3/P5 accepted region 做 conditional bad-event autopsy。
3. 训练/校准 legal conditional bad-event statistics。
4. 重构 safe-useful target，不再把 useful 与 safe 分开做弱串联。
5. 在 CBD0 exact reference 上恢复 deployable frontier。
6. 并行继续 true-delta compute v8，但不让 compute lane 掩盖 decision blocker。
```

v9.2.63 的核心 stop-go 结论必须是以下之一：

```text
Case 1:
  conditional bad-event 可分，reference frontier deployable。
  → 继续 true-delta compute / system controller / LDO / paired replay。

Case 2:
  conditional bad-event 可分，但 true-delta compute 仍不 system-legal。
  → decision geometry viable，kernelization 是主 blocker。

Case 3:
  conditional bad-event 不可分。
  → current exact branch-delta signal 仍只能 diagnostic，必须重设 output-delta sufficient statistics。

Case 4:
  reference frontier point estimate pass，但 LCB/UCB fail。
  → sample confidence / family stability / calibration still blocker。

Case 5:
  support stability regression。
  → 先修 generator，不允许 official promotion。
```

---

# Part III. 核心假设

## H1：v9.2.62 的 main failure 是 useful-positive conditional bad-event，而不是 null contamination

证据：

```text
P2 best null rejector:
  null AUC = 0.720924
  null rate after gate = 0.139792

P5 best frontier:
  null_rate = 0.020101
  bad-event = 0.236181
```

H1 成立标准：

在 P5-style null-rejected region 内，conditional bad-event classifier 达到：

$$
AUC_{\text{bad} \mid U,\neg N,S}\geq0.70
$$

or:

$$
Corr(S_{\text{bad-cond}},Y_{\text{bad}}\mid U,\neg N,S)\geq0.35.
$$

并且加入 conditional bad gate 后：

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

同时：

$$
Coverage_{\text{heldout}}\geq0.03.
$$

H1 失败标准：

conditional bad-event AUC 低于 `0.60`，或 bad-event 降低时 coverage 回到 tiny slice。

## H2：useful score 需要 risk-adjusted，而不是 sequential useful ∧ risk gate

v9.2.62 的 useful score能找到 coverage，但高 risk。H2 认为必须使用风险调整后的 value lower bound：

$$
S_{\text{safe-useful}}(e)
=
LCB_{\text{useful}}(e)
-
\lambda_B UCB_{\text{bad}}(e)
-
\lambda_N P_{\text{null}}(e).
$$

H2 成立标准：

risk-adjusted score 在 exact reference heldout 上达到：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H2 失败标准：

risk-adjusted score仍只能在 coverage `<0.02` 时安全，或 coverage 过线时 bad-event > `0.05`。

## H3：bad-event 在 useful-positive region 有不同于 general bad-event 的子模式

H3 认为 v9.2.62 的 bad-event 不是一般 risk，而是 useful-positive 条件下的特殊风险：

```text
B1-useful-tail-harm:
  useful score 高，但 CEp99 / wrong-confidence tail 不稳定。

B2-useful-control-fragile:
  functional gain 高，但 controls 在邻近 family 中同样可达或更稳定。

B3-useful-delta-overreach:
  delta/logit movement 大，短期 gain 高，泛化或 tail 风险高。

B4-useful-family-unstable:
  family reliability 点估计高，但 LFO / LSO reliability 不稳。

B5-useful-horizon-inconsistent:
  h20/h80 有 gain，h240/h640 出现 harm 或消失。

B6-useful-support-edge:
  位于 support boundary，density enough but confidence weak。
```

H3 成立标准：

至少三个 conditional bad submodes 有 legal statistic 达到：

$$
AUC_{\text{submode}}\geq0.65
$$

or:

$$
Corr_{\text{submode}}\geq0.30.
$$

H3 失败标准：

submode statistics 均无信号，或只能用 posthoc replay 才能分离。

## H4：support stability 已经过关，本轮应把 support 用作 confidence bound，而不是继续 blind expansion

v9.2.62 support：

```text
signal strata = 27
families = 741
accepted families = 71
accepted support pass = 1
```

H4 成立标准：

v9.2.63 support audit 保持：

```text
natural_real_event_count >= 24000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 20
measured_family_count >= 700
duplicate_row_count = 0
```

accepted set：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

H4 失败标准：

support generator 回退，或 accepted region 又退回 single-stratum / few-family pocket。

## H5：true-delta compute 仍是并行 blocker，但不是本轮唯一主 blocker

P6 仍是：

```text
AUC = 0.879072
agreement = 1.0
step ratio q90 = 2.863280
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

## 1. 从三类改成四类

v9.2.62 的三类：

```text
safe-good
harmless-null
bad-event
```

不足以解释 useful-positive bad-event。v9.2.63 改为四类：

```text
safe-useful:
  useful and safe.

risky-useful:
  useful-looking but bad-event.

harmless-null:
  safe but not useful.

bad-null:
  bad and not useful.
```

形式化：

$$
Y_{\text{useful}}(e)\in\{0,1\},
$$

$$
Y_{\text{bad}}(e)\in\{0,1\},
$$

$$
Y_{\text{null}}(e)\in\{0,1\}.
$$

四类定义：

$$
Y_{\text{safe-useful}}
=
Y_{\text{useful}}
\land
(1-Y_{\text{bad}})
\land
(1-Y_{\text{null}}).
$$

$$
Y_{\text{risky-useful}}
=
Y_{\text{useful}}
\land
Y_{\text{bad}}.
$$

$$
Y_{\text{harmless-null}}
=
(1-Y_{\text{useful}})
\land
(1-Y_{\text{bad}}).
$$

$$
Y_{\text{bad-null}}
=
(1-Y_{\text{useful}})
\land
Y_{\text{bad}}.
$$

## 2. Conditional bad-event target

本轮新增最重要标签：

$$
Y_{\text{bad} \mid U,\neg N,S}
=
Y_{\text{bad}}
\quad
\text{conditioned on}
\quad
UsefulPositive \land NullRejected \land SupportStable.
$$

它不是 posthoc commit rule。它用于校准 legal sufficient statistics。Official commit 只能用 train-stream features。

## 3. Risk-adjusted safe-useful score

旧 sequential gate：

$$
Accept = UsefulPositive \land NullRejected \land RiskSafe \land SupportStable.
$$

v9.2.63 引入风险调整分数：

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

Accept：

$$
Accept(e)
=
\mathbb{1}[S_{\text{safe-useful}}(e)\geq\tau_S]
\land
\mathbb{1}[UCB_{\text{bad-cond}}(e)\leq\rho_B]
\land
\mathbb{1}[P_{\text{null}}(e)\leq\rho_N]
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
NullRate(A)\leq0.10,
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

## 1. Conditional bad-event statistics

### CB0：v9.2.62 R2 reference

Reference only.

### CB1：UsefulTailHarmUCB

Tail harm in useful-positive region:

$$
S_{\text{tail-harm}}
=
UCB(CEp99)
+
\alpha_1 UCB(WrongConfP95)
-
\alpha_2 LCB(MarginP10)
+
\alpha_3 Volatility_{\text{tail}}.
$$

### CB2：UsefulControlFragility

Measures whether useful gain is fragile relative to matched controls:

$$
S_{\text{control-fragile}}
=
UCB(Gain_{\text{controls}})
-
LCB(Gain_F)
+
\alpha Var_f(Gap).
$$

### CB3：DeltaOverreachRisk

Large movement with unstable gain:

$$
S_{\text{overreach}}
=
\frac{UCB(\|\Delta z_F\|)}
{LCB(Gain_F-\max Gain_{\text{controls}})+\epsilon}
+
\beta
\frac{UCB(\|\Delta\theta_F\|)}
{LCB(Gap)+\epsilon}.
$$

### CB4：HorizonInconsistencyRisk

Useful short-horizon signal that fails later:

$$
S_{\text{horizon-risk}}
=
\max(0, Gap_{20})
+
\max(0, Gap_{80})
-
\min(Gap_{240},Gap_{640}).
$$

High value means short-term positive but long-term weak/harmful.

### CB5：FamilyInstabilityUCB

Family reliability upper bad-event bound:

$$
UCB_{\text{bad}}(f)
=
\hat{p}_{bad}(f)
+
\kappa
\sqrt{
\frac{\hat{p}_{bad}(f)(1-\hat{p}_{bad}(f))}{n_f+\epsilon}
}.
$$

### CB6：ConditionalBadMixture

Monotone mixture:

$$
UCB_{\text{bad-cond}}
=
b_1S_{\text{tail-harm}}
+
b_2S_{\text{control-fragile}}
+
b_3S_{\text{overreach}}
+
b_4S_{\text{horizon-risk}}
+
b_5UCB_{\text{bad}}(f).
$$

Coefficients selected on calibration split only; no dataset name.

---

## 2. Value/control statistics

### V0：v9.2.62 U1/U6 reference

Reference only.

### V1：ValueControlLCBv3

$$
LCB_{\text{value-control}}
=
LCB(Gain_F)
+
LCB(Gain_F-\max Gain_{\text{controls}}).
$$

### V2：SafeUsefulMargin

Direct margin between safe-useful and other classes:

$$
S_{\text{margin}}
=
S_{\text{useful}}
-
\max(
S_{\text{risky-useful}},
S_{\text{null}},
S_{\text{bad-null}}
).
$$

### V3：PersistentNoRegretValue

No-regret value across controls and horizons:

$$
S_{\text{persist-noregret}}
=
\min_{h\in\mathcal{H}}
[
Gain_{F,h}
-
\max_{c\in\mathcal{C}}Gain_{c,h}
].
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

### V4：EfficientSafeGain

$$
S_{\text{eff-safe}}
=
\frac{
LCB(Gap)
}{
UCB(\|\Delta z_F\|)+\epsilon
}
-
\lambda_B UCB_{\text{bad-cond}}.
$$

---

## 3. Null statistics

### N0：v9.2.62 N2 reference

Reference only.

### N1：LowGainNullLCB

$$
P_{\text{null-lowgain}}
=
\sigma
(
-\alpha LCB(Gain_F)
).
$$

### N2：ControlEquivalentNullV2

$$
P_{\text{null-control}}
=
\sigma
(
-\alpha LCB(Gain_F-\max Gain_{\text{controls}})
).
$$

### N3：DeltaSilentNullV2

$$
P_{\text{null-silent}}
=
\sigma
(
-\alpha
\frac{\|\Delta z_F\|}
{\operatorname{median}(\|\Delta z_F\|)+\epsilon}
).
$$

### N4：HybridNullProbabilityV2

$$
P_{\text{null}}
=
1-\prod_m(1-P_{\text{null},m}).
$$

---

## 4. Support statistics

### S0：v9.2.62 support reference

Reference only.

### S1：ConditionalFamilyReliability

Family must not include dataset name:

$$
family(e)
=
(
stratum,
horizon,
role\_bucket,
carrier,
value\_bucket,
null\_bucket,
conditional\_bad\_bucket,
control\_gap\_bucket,
tail\_risk\_bucket,
delta\_overreach\_bucket
).
$$

Family reliability:

$$
Rel_{\text{safe-useful}}(f)
=
LCB[
P(Y_{\text{safe-useful}}=1\mid f)
].
$$

### S2：LeaveConditionFamilyOutReliability

$$
Rel_{\text{LCFO}}(f)
=
\min_{f'\in\mathcal{N}(f)}
Rel_{\text{safe-useful}}(f').
$$

### S3：RiskAdjustedKNNPocket

Feature vector:

```text
LCB value-control
conditional bad UCB
null probability
tail harm score
control fragility score
delta overreach score
family reliability
support density
horizon bucket
stratum bucket
role bucket
```

Density:

$$
Density(e)=
\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### S4：CoverageBalancedSupportCapV2

Accepted set constraints:

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

---

# Part VI. Controller candidates

## C0：v9.2.62 best reference

Reference only.

## C1：ConditionalBadGateController

$$
Accept(e)
=
UsefulPositive(e)
\land
NullRejected(e)
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
SupportStable(e).
$$

## C2：RiskAdjustedSafeUsefulController

$$
Accept(e)
=
S_{\text{safe-useful}}(e)\geq\tau_S
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
P_{\text{null}}(e)\leq\rho_N
\land
SupportStable(e).
$$

## C3：FourClassMarginController

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

## C4：PersistentSafeUsefulController

$$
Accept(e)
=
S_{\text{persist-noregret}}(e)\geq\tau_P
\land
UCB_{\text{bad-cond}}(e)\leq\rho_B
\land
P_{\text{null}}(e)\leq\rho_N
\land
Rel_{\text{LCFO}}(family(e))\geq r_0.
$$

## C5：ConstrainedRiskAdjustedOptimizer

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
NullRate(A)\leq0.10,
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

## P0：v9.2.62 boundary reproduction

### 目标

确认 v9.2.62 boundary 稳定，避免基于偶然 P5 frontier 继续推进。

### 必须记录

```text
route
source_route_v9261
useful_autopsy_pass
null_rejector_pass
useful_stat_pass
support_stability_pass
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
true_delta_compute_pass
true_delta_step_ratio_q90
oracle_support_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R12-UsefulTargetStillTiny
support_stability_pass = 1
exact_reference_deployable = 0
true_delta_compute_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_v9261_to_v9262_progress_ladder.svg
p0_null_vs_bad_event_tradeoff.svg
```

---

## P1：conditional bad-event autopsy inside accepted regions

### 目标

把 P3/P5 accepted rows 分解，确认 bad-event 是在 useful-positive/null-rejected/support-stable 条件下发生，还是某个子 gate 失控。

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
useful_positive
null_rejected
support_stable
safe_useful
risky_useful
harmless_null
bad_null
bad_event
tail_harm
control_fragility
delta_overreach
horizon_inconsistency
family_instability
gap_score
useful_score
null_score
bad_score
support_score
```

Failure modes：

```text
B1-useful-tail-harm
B2-useful-control-fragile
B3-useful-delta-overreach
B4-useful-family-unstable
B5-useful-horizon-inconsistent
B6-useful-support-edge
N1-harmless-null-leak
S1-support-confidence-fail
V1-useful-score-overbroad
```

### 判断标准

P1 pass：

```text
conditional_bad_event_attribution_fraction >= 0.90
risky_useful_attribution_fraction >= 0.90
harmless_null_attribution_fraction >= 0.90
```

并且必须报告：

```text
P(bad | useful_positive)
P(bad | useful_positive, null_rejected)
P(bad | useful_positive, null_rejected, support_stable)
P(safe_useful | useful_positive, null_rejected, support_stable)
```

### 可视化

```text
p1_four_class_sankey.svg
p1_conditional_bad_event_rates.svg
p1_bad_event_mode_breakdown.svg
p1_useful_score_vs_bad_event_scatter.svg
p1_p5_accepted_region_decomposition.svg
```

---

## P2：conditional bad-event statistic factory

### 目标

建立 legal online conditional bad-event predictor，专门分离 safe-useful 与 risky-useful。

### 必须记录

```text
bad_stat_id
conditional_region
bad_submode
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_conditional_bad
corr_conditional_bad
AUC_submode
bad_event_after_gate
coverage_after_gate
precision_after_gate
null_rate_after_gate
calibration_error
feature_overhead
memory_overhead
```

### 判断标准

Conditional bad statistic pass：

$$
AUC_{\text{bad} \mid U,\neg N,S}\geq0.70
$$

or:

$$
Corr_{\text{bad} \mid U,\neg N,S}\geq0.35.
$$

Submode diagnostic pass：

At least three submodes satisfy:

$$
AUC_{\text{submode}}\geq0.65
$$

or:

$$
Corr_{\text{submode}}\geq0.30.
$$

Utility pass：

$$
BadEventRate\leq0.05,
$$

$$
Coverage\geq0.03,
$$

$$
Precision\geq0.75.
$$

### 可视化

```text
p2_conditional_bad_auc_matrix.svg
p2_conditional_bad_reduction_curve.svg
p2_submode_risk_frontier.svg
p2_bad_calibration_curve.svg
```

---

## P3：risk-adjusted safe-useful score factory

### 目标

用 LCB/UCB 和 conditional bad UCB 构造 safe-useful score，避免 useful score 单独选择 risky rows。

### 必须记录

```text
safe_useful_score_id
value_stat_id
conditional_bad_stat_id
null_stat_id
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

Safe-useful score pass：

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
NullRate_{\text{heldout}}\leq0.10.
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
p3_safe_useful_score_frontier.svg
p3_value_bad_null_tradeoff_surface.svg
p3_lcb_ucb_constraint_curve.svg
p3_safe_useful_ablation.svg
```

---

## P4：support confidence and family stability audit

### 目标

确认 support stability 继续过关，并将 support 用作 confidence bound，而不是盲目扩 support。

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
conditional_bad_ucb_family
safe_useful_lcb_family
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
measured_signal_strata_count >= 20
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
p4_support_stability_heatmap.svg
p4_conditional_family_bad_ucb.svg
p4_accepted_family_balance.svg
p4_signal_strata_coverage.svg
```

---

## P5：exact reference deployable frontier v7

### 目标

组合 P2/P3/P4，在 CBD0 exact reference 上判断是否存在 deployable safe-useful frontier。

### 必须记录

```text
controller_id
safe_useful_score_id
conditional_bad_stat_id
null_stat_id
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
NullRate_{\text{heldout}}\leq0.10.
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

If no controller passes, route must be:

```text
R10-ConditionalBadEventInseparable
```

or:

```text
R11-ReferenceStillNotDeployableAfterConditionalRisk
```

depending on P2 outcome.

### 可视化

```text
p5_reference_frontier_v7_precision_coverage_bad_null.svg
p5_conditional_bad_gate_surface.svg
p5_safe_useful_deployable_region.svg
p5_oracle_legal_gap_after_conditional_risk.svg
```

---

## P6：true-delta compute v8 parallel lane

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
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
safe_useful_component_time
conditional_bad_component_time
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
p6_true_delta_v8_cost_signal_pareto.svg
p6_compute_vs_decision_ladder.svg
p6_residual_subphase_after_conditional_risk.svg
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
conditional_bad_stat_id
null_stat_id
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
NullRate_{\text{heldout}}\leq0.10,
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
conditional_bad_stat_id
null_stat_id
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
conditional_bad_stat_id = best P8 survivor
null_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledSafeUsefulScore, ShuffledConditionalBad,
           ShuffledNullRejector, ShuffledSupportStat, ShuffledControlGain,
           ShuffledCandidateGate, ShuffledBranchRatio, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
safe_useful_score_id
conditional_bad_stat_id
null_stat_id
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
ShuffledConditionalBad = fail
ShuffledNullRejector = fail
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
contract_audit_v9263.csv
p0_v9262_boundary_reproduction.csv
p1_conditional_bad_event_autopsy.csv
p2_conditional_bad_event_stat_factory.csv
p3_risk_adjusted_safe_useful_score_factory.csv
p4_support_confidence_family_stability_audit.csv
p5_exact_reference_deployable_frontier_v7.csv
p6_true_delta_compute_v8_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
conditional_bad_trace_v9263.csv
safe_useful_score_trace_v9263.csv
support_confidence_trace_v9263.csv
reference_frontier_v7_trace.csv
true_delta_compute_v8_trace.csv
system_controller_trace_v9263.csv
leaveout_trace_v9263.csv
paired_replay_branch_trace_v9263.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9262_boundary_unstable
F3_dataset_tuning_detected
F4_conditional_bad_autopsy_incomplete
F5_conditional_bad_event_inseparable
F6_conditional_bad_submode_unattributed
F7_safe_useful_score_not_predictive
F8_safe_useful_score_tiny_coverage
F9_bad_event_ucb_fail
F10_precision_lcb_fail
F11_support_stability_regression
F12_support_balance_fail
F13_exact_reference_still_not_deployable_after_conditional_risk
F14_true_delta_compute_still_expensive
F15_true_delta_signal_lost
F16_system_controller_precision_fail
F17_system_controller_coverage_fail
F18_system_controller_bad_event_fail
F19_system_controller_null_rate_fail
F20_leave_dataset_out_fail
F21_leave_stratum_out_fail
F22_paired_replay_control_equivalent
F23_shuffle_control_pass
F24_functional_lr_equivalent
F25_short_run_task_drop
F26_full_run_no_macro_hard_stratum_gain
F27_strong_baseline_explains_gain
F28_robustness_fail
F29_external_not_ready
F30_fake_or_proxy_violation
F31_artifact_missing
```

---

# Part IX. Route decision

```text
R1-BoundaryReproduced:
  v9.2.62 boundary is reproduced.

R2-ConditionalBadAutopsyPass:
  conditional bad-event modes in accepted regions are attributed.

R3-ConditionalBadStatisticPass:
  conditional bad-event predictor separates risky-useful from safe-useful.

R4-SafeUsefulScorePass:
  risk-adjusted safe-useful score is predictive and coverage-preserving.

R5-SupportConfidencePass:
  support remains broad and accepted set remains balanced.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + conditional bad / safe-useful frontier passes precision / coverage / bad-event / null gate.

R7-TrueDeltaComputeV8SystemPass:
  true branch-delta v8 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-ConditionalBadEventInseparable:
  useful-positive / null-rejected region contains bad-events without legal predictable signature.

R13-SafeUsefulStillTiny:
  safe-useful score can be safe only at coverage < 0.03.

R14-ReferenceFeasibleButComputeFail:
  decision geometry viable, true-delta compute still too expensive.

R15-ComputePassButReferenceFail:
  compute viable, accept-region geometry still fails.

R16-SupportRegression:
  support stability regresses and blocks official promotion.

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
v9262_boundary_pass
dataset_tuning_detected
conditional_bad_autopsy_pass
conditional_bad_attribution_fraction
risky_useful_attribution_fraction
harmless_null_attribution_fraction
best_conditional_bad_stat_id
conditional_bad_stat_pass
conditional_bad_auc
conditional_bad_corr
conditional_bad_after_gate
coverage_after_conditional_bad_gate
best_safe_useful_score_id
safe_useful_score_pass
safe_useful_auc
safe_useful_precision
safe_useful_coverage
safe_useful_bad_event
safe_useful_null_rate
precision_lcb
bad_event_ucb
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
success_v9263_strict_purekan_functional
success_v9263_full_functional
success_v9263_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 conditional bad-event autopsy
  P2 conditional bad-event statistic factory
  P3 risk-adjusted safe-useful score factory
  P4 support confidence audit
  P5 exact reference deployable frontier v7
  P6 true-delta compute v8

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
v9.2.62 boundary reproduced
conditional bad-event autopsy completed
conditional bad-event statistic measured
safe-useful score measured
support confidence audited
exact reference deployable frontier v7 measured
true-delta compute v8 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
conditional bad statistic pass
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
1. v9.2.62 boundary cannot be reproduced；
2. conditional bad-event autopsy cannot attribute risky-useful rows；
3. conditional bad-event has no legal predictable signature；
4. safe-useful score remains tiny coverage；
5. coverage crosses 0.03 but bad-event UCB remains above 0.05；
6. precision LCB remains below 0.75；
7. support stability regresses；
8. exact reference remains not deployable after conditional-risk repair；
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

下一步继续 kernel / layout / fused safe-useful / conditional-risk compute，不调 dataset。

## Case C：compute pass but reference frontier fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 safe-useful / conditional-bad sufficient statistics，不走 kernel-only。

## Case D：conditional bad-event inseparable

必须声明：

```text
useful-positive region contains bad-events that current legal features cannot separate.
```

下一步重设 output-delta sufficient statistics or safe-useful target semantics。

## Case E：safe-useful score remains tiny

必须声明：

```text
controller can find safe rows, but cannot recover deployable coverage.
```

下一步不是继续阈值小调，而是重新设计 value/risk representation。

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

v9.2.63 的一句话策略是：

$$
\boxed{
\text{不要再把 bad-event 当作全局 risk；现在要在 useful-positive/null-rejected/support-stable 条件下建模 conditional bad-event。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
support generator 是否过；
null rejector 是否能压 null；
useful score 是否有 AUC；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. P3/P5 accepted bad-events 到底是哪类 useful-positive failure？
2. 在 useful-positive / null-rejected / support-stable 条件下，bad-event 是否可预测？
3. tail-harm / control-fragility / delta-overreach / horizon inconsistency / family instability 哪个是主因？
4. 能否用 conditional bad UCB 构造 risk-adjusted safe-useful score？
5. exact-reference coverage 能否从 0.016452 拉到 >=0.03？
6. bad-event 能否从 0.236181 压到 <=0.05？
7. bad-event UCB 能否从 0.280308 压到 <=0.05？
8. accepted set 能否保持 >=5 strata / >=32 families？
9. true delta compute v8 能否继续压到 step <=1.50？
10. system-legal controller 能否 LDO/LSO？
11. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.63 的结果将给出清晰分叉：

```text
if conditional bad risk separates risky-useful and reference frontier becomes deployable:
  decision geometry is viable; proceed to true-delta compute and official controller.

if conditional bad risk separates but compute fails:
  kernelization remains blocker.

if conditional bad risk cannot separate:
  current exact branch-delta signal remains diagnostic; redesign output-delta sufficient statistics.

if reference frontier point estimate passes but confidence fails:
  improve calibration/support confidence, not dataset-specific tuning.

if both reference and compute pass:
  open LDO/LSO and official paired replay.
```
