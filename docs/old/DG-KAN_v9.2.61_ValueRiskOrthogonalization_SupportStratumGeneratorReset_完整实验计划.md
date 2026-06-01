# DG-KAN v9.2.61 Value-Risk Orthogonalization 与 Support-Stratum Generator Reset 完整实验计划

> 本计划基于 v9.2.60 `Clean-Core Expansion 与 Constraint-Calibrated Support Frontier` 的真实复盘制定。  
> v9.2.60 的 terminal route 是：
>
> ```text
> route = R12-CleanCoreNotExpandable
> base_candidate = LQ-t2-h256
> success_v9260_strict_purekan_functional = False
> success_v9260_full_functional = False
> success_v9260_external_ready = False
> ```
>
> v9.2.60 的关键事实是：
>
> ```text
> P1 clean-core / near-core:
>   core_precision = 0.815789
>   core_coverage = 0.007540
>   core_bad_event = 0.085526 > 0.05
>   clean_core_stable = 0
>
>   near_core_oracle_overlap = 0.134910
>   near_core_safe_good_density = 0.134910
>   global_safe_good_density = 0.130060
>   near_core_bad_event = 0.155563
>   nearcore_expandable = 0
>
> P2 support measurement:
>   natural_real_event_count = 20160
>   balanced_diagnostic_real_event_count = 591 < 6000
>   measured_signal_strata_count = 3 < 6
>   measured_family_count = 606
>   duplicate_row_count = 0
>   support_measurement_pass = 0
>
> P3 risk-budget frontier:
>   best = RB3-TailFirstSoftBudget
>   precision = 0.186069
>   coverage = 0.051984
>   bad-event = 0.043893
>   diagnostic = 1
>   official pass = 0
>
> P4 support frontier:
>   best = SF4-DensityRiskJointPocket
>   precision = 0.960784
>   coverage = 0.002530
>   bad-event = 0.0
>   strata = 1
>   pass = 0
>
> P5 exact reference deployable frontier:
>   best = C-RB4-ControlGapConditionalRisk + SF4-DensityRiskJointPocket
>   precision = 0.956522
>   coverage = 0.003423
>   bad-event = 0.014493
>   precision LCB = 0.879786
>   bad-event UCB = 0.077633 > 0.05
>   strata = 1
>   exact_reference_deployable = 0
>
> P6 true-delta compute:
>   AUC = 0.879914
>   agreement = 1.0
>   step ratio = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> oracle:
>   precision = 1.0
>   coverage = 0.119978
>   bad-event = 0.0
>   oracle_support_pass = 1
>
> current blocker:
>   clean_core_not_stable
>
> next_required_implementation:
>   recalibrate_clean_core_definition
> ```
>
> v9.2.61 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.60 证明 clean-core expansion 这条路线不成立；下一步不能继续扩张一个不稳定的 core。}
> }
> $$
>
> 这不是 functional update 失败，也不是 exact branch-delta signal 消失。相反，v9.2.60 证明了三个更细的事实：
>
> $$
> \boxed{
> \text{risk 可以控制 bad-event，但不保证 value precision。}
> }
> $$
>
> $$
> \boxed{
> \text{support 可以找到高 precision pocket，但 coverage / strata 太小。}
> }
> $$
>
> $$
> \boxed{
> \text{clean core 本身在 fresh run 中 bad-event 超线，因此不能作为 expansion seed。}
> }
> $$
>
> 因此 v9.2.61 的主线必须从 **clean-core expansion** 改为 **value-risk orthogonalization + support-stratum generator reset**。  
> 换句话说，本轮不再问：
>
> ```text
> 这个 clean core 能不能继续扩张？
> ```
>
> 而是问：
>
> ```text
> 什么 legal online statistics 能同时区分：
>   safe-good；
>   harmless-but-not-useful；
>   bad-event？
>
> 并且怎样真实生成足够多 signal strata / family support，
> 让 controller 不再依赖单一 narrow pocket？
> ```

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
  report MNIST / Fashion-MNIST / KMNIST slice failure modes
  report bad-event by dataset
  report value-miss by dataset
  report support density by dataset
  report leave-dataset-out matrices
  diagnose whether a dataset concentrates one failure mode
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: use threshold A
  if dataset == KMNIST: use risk statistic B
  if dataset == MNIST: skip branch-delta
  tune thresholds per dataset
  choose support family per dataset
  choose candidate generator per dataset
  choose selected/full-logit mode separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official controller 只能使用 train-stream / event-level / role-level / value-risk-support-family-level / branch-delta features。Dataset name 只能出现在 report 中，不能出现在 commit rule 中。

---

# Part I. 对 v9.2.60 的独立判断

## 1. v9.2.60 没有达到目标

v9.2.60 没有 strict PureKAN functional success。失败发生在前置 decision gate 与 compute gate：

```text
clean_core_stable = 0
nearcore_expandable = 0
support_measurement_pass = 0
risk_budget_pass = 0
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

特别要注意：P5 的 best exact-reference frontier 看上去有很高 precision：

```text
precision = 0.956522
bad-event = 0.014493
coverage = 0.003423
```

但它仍不能写成 near-pass，因为：

```text
coverage < 0.03
bad-event UCB = 0.077633 > 0.05
accepted signal strata = 1
```

这不是可部署 controller，而是一个 narrow high-confidence pocket。

## 2. v9.2.60 的真实进展

v9.2.60 仍有三项进展。

第一，clean-core hypothesis 被真正检验并否定。v9.2.59 的 tiny clean core 在 v9.2.60 fresh run 中不稳定：

$$
BadEvent(Core)=0.085526>0.05.
$$

这说明 v9.2.59 的 core 不是一个可以直接扩张的稳定种子。

第二，risk-budget frontier 发现了一个重要 diagnostic region：

```text
RB3-TailFirstSoftBudget:
  coverage = 0.051984
  bad-event = 0.043893
  precision = 0.186069
```

这说明 risk 可以把 bad-event 控制在 official safety gate 附近，并且 coverage 不再 tiny。但 precision 极低，意味着它接收了大量 **not-bad but not-useful** 的 rows。这个结果把问题从 “risk 是否可预测” 推进到 “value 和 risk 必须正交建模”。

第三，support measurement 的 natural rows 与 family count 已经上来了：

```text
natural_real_event_count = 20160
measured_family_count = 606
duplicate_row_count = 0
```

但 balanced diagnostic rows 与 signal strata 仍不足：

```text
balanced_diagnostic_real_event_count = 591
measured_signal_strata_count = 3
```

这说明当前 row generator 不是造假或复制问题，而是 **signal strata 生成机制不足**。

## 3. v9.2.60 的真实失败

v9.2.60 的失败不是一个 threshold 没调好，而是三个结构性问题同时存在。

### 3.1 Clean core 不稳定

v9.2.59 的 clean core 被当作 expansion seed，但 v9.2.60 的 core：

```text
precision = 0.815789
coverage = 0.007540
bad-event = 0.085526
```

核心问题是 bad-event 已经超过 gate。也就是说，不能继续围绕这个 core 做 near-core expansion。

### 3.2 Near-core 不富集 oracle support

Near-core 的 oracle overlap / safe-good density 只有：

$$
0.134910.
$$

全局 safe-good density 是：

$$
0.130060.
$$

差值只有：

$$
0.134910-0.130060=0.004850.
$$

计划要求 near-core 至少比 global 高 `+0.10`。当前 near-core 与随机全局相比几乎没有 enrichment，所以 near-core 不是 oracle pocket。

### 3.3 Risk 与 value 混在一起，导致两类错误

P3 risk-budget 的 `RB3` 证明：

```text
risk 可以压 bad-event；
coverage 可以达到 0.051984；
但 precision 只有 0.186069。
```

这说明 accepted rows 大多不是 bad-event，但也不是 safe-good。  
它们更像是：

```text
harmless-null
control-equivalent
low-value
non-control-resistant
```

因此当前 controller 不是简单缺 risk，而是缺一个能和 risk 分离的 **value-positive / control-resistant gate**。

## 4. 当前 blocker 的本质

当前 blocker 应从：

```text
clean_core_not_stable
```

进一步解释为：

$$
\boxed{
\text{value-risk-support 三者没有被正交化，且 support-stratum generator 仍不足。}
}
$$

也就是：

```text
1. exact branch-delta ranking signal 仍强；
2. oracle support 仍存在；
3. risk 能找到安全但低价值区域；
4. support 能找到高精度但 tiny/单 stratum 区域；
5. clean core 不稳定，不能继续扩张；
6. signal strata 和 balanced diagnostic support 仍不足；
7. true-delta compute 仍太贵。
```

所以 v9.2.61 必须同时做两件事：

```text
A. 重建 controller 的 decision variables：
   value-positive / control-resistant / risk-safe / support-stable 分离建模。

B. 重建 event/support generator：
   不再只在现有 3 个 signal strata 上找 pocket，
   而是主动生成更均衡的 train-stream event family。
```

## 5. 是否还在正确道路上

是，但要换主线。

正确路线现在是：

```text
exact signal exists
→ oracle support exists
→ clean-core expansion failed
→ value-risk orthogonalization
→ support-stratum generator reset
→ exact reference deployable frontier
→ true-delta system path
→ LDO / LSO
→ paired replay
```

错误路线是：

```text
继续围绕不稳定 clean core 扩张；
继续把所有 risk modes 做 hard union；
继续只优化 true-delta kernel；
继续调 C0/C6 threshold；
把 RB3 的 coverage/bad-event 写成 success；
把 SF4 的 high precision pocket 写成 success；
按 MNIST/Fashion/KMNIST 分别调 gate。
```

---

# Part II. v9.2.61 总体目标

v9.2.61 的总体目标是：

$$
\boxed{
\text{建立 value-risk-support 正交 controller，并通过 support-stratum generator reset 生成可部署 exact-reference frontier。}
}
$$

具体目标分为八层：

```text
1. 复现 v9.2.60 boundary，确认 clean core not expandable。
2. 分解 accepted rows 为 safe-good / harmless-null / bad-event。
3. 构造 legal value-positive / control-resistant sufficient statistics。
4. 构造 legal risk-safe sufficient statistics，但不让 risk 独自决定 accept。
5. 重置 support-stratum event generator，使 balanced diagnostic rows 与 signal strata 达标。
6. 在 CBD0 exact reference 上验证 value-risk-support frontier 是否可部署。
7. 并行继续 true-delta compute v6。
8. 如果 reference deployable + compute pass，同时打开 LDO/LSO 与 paired replay scout。
```

本轮最重要的 stop-go 问题是：

$$
\boxed{
\text{RB3 的安全 coverage 能否被 value gate 筛成 high precision region？}
}
$$

如果不能，则说明 branch-delta exact signal 与当前 legal features 仍无法区分 harmless-null 和 useful safe-good。此时继续 kernelization 没意义。

---

# Part III. 核心假设

## H1：v9.2.60 的 RB3 失败主要是 value failure，不是 risk failure

证据：

```text
RB3 coverage = 0.051984
RB3 bad-event = 0.043893
RB3 precision = 0.186069
```

H1 认为 RB3 已经找到 safety-feasible region，但该 region 混入大量 low-value / control-equivalent rows。

H1 成立标准：

在 RB3 candidate region 内加入 legal value-positive / control-resistant gate 后：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\geq0.03,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H1 失败标准：

RB3 region 内没有任何 legal value gate 能把 precision 提到 `0.75`，或者提到 `0.75` 后 coverage 低于 `0.01`。

## H2：safe-good 应该拆为 value-positive 与 control-resistant，而不是继续用单一 precision label

定义：

$$
Y_{\text{value}}(e)
=
\mathbb{1}
[
Gain_{\text{RealFunctional}}(e)>0
].
$$

$$
Y_{\text{control}}(e)
=
\mathbb{1}
[
Gain_{\text{RealFunctional}}(e)
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
>
\gamma_c
].
$$

$$
Y_{\text{useful}}(e)
=
Y_{\text{value}}(e)
\land
Y_{\text{control}}(e).
$$

H2 成立标准：

至少一个 legal value/control statistic 达到：

$$
AUC(Y_{\text{useful}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},Y_{\text{useful}})\geq0.35.
$$

并且加入 controller 后：

$$
Precision_{\text{safe-good}}\geq0.75.
$$

H2 失败标准：

所有 legal value/control statistics 都低于 AUC `0.60`，或无法提升 precision。

## H3：support frontier 失败来自 signal-stratum generation 不足，而不只是 family scorer 错

v9.2.60 已有：

```text
natural_real_event_count = 20160
measured_family_count = 606
```

但：

```text
balanced_diagnostic_real_event_count = 591
measured_signal_strata_count = 3
```

H3 认为当前不是 family 数不够，而是 signal strata 的 event generator 没覆盖到足够多 functional modes。

H3 成立标准：

v9.2.61 的 support generator 达到：

```text
natural_real_event_count >= 24000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 32
duplicate_row_count = 0
```

并且 exact reference frontier 的 accepted strata：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

H3 失败标准：

natural rows 继续足够但 measured signal strata 仍 `<=3`，说明 event generator 本身没有打开足够 modes。

## H4：clean-core seed 已经失效，下一步应使用 value-risk-support frontier，而不是 core-neighbor expansion

H4 成立标准：

不依赖 v9.2.59/v9.2.60 clean core seed 的 frontier candidate 通过：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H4 失败标准：

所有非-core frontier 仍 fail，而 core-based frontier 也 fail。则 exact signal 的 current sufficient statistics 不足，需要重设 branch-delta output statistics。

## H5：true-delta compute 仍是并行 blocker，但不是 v9.2.61 的唯一主 blocker

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

true-delta compute 仍：

$$
StepRatio_{q90}>1.50
$$

or signal lost。

但如果 H1-H4 失败，即使 H5 成立，也不能进入 functional success。

---

# Part IV. New decision model

## 1. 三类标签

v9.2.61 将 accepted rows 显式分成三类：

```text
safe-good:
  useful and safe

harmless-null:
  safe but not useful, control-equivalent, value-low, or no causal gain

bad-event:
  unsafe / tail-harm / bad generalization event
```

形式化：

$$
Y_{\text{bad}}(e)\in\{0,1\}.
$$

$$
Y_{\text{useful}}(e)=Y_{\text{value}}(e)\land Y_{\text{control}}(e).
$$

$$
Y_{\text{safe-good}}(e)
=
Y_{\text{useful}}(e)
\land
(1-Y_{\text{bad}}(e)).
$$

$$
Y_{\text{harmless-null}}(e)
=
(1-Y_{\text{useful}}(e))
\land
(1-Y_{\text{bad}}(e)).
$$

v9.2.60 的 RB3 失败可以解释为：

$$
Y_{\text{bad}}\text{ low, but }Y_{\text{useful}}\text{ also low.}
$$

## 2. Controller decomposition

Official controller 不再由一个 mixed score 决定，而是：

$$
Accept(e)
=
ValuePositive(e)
\land
ControlResistant(e)
\land
RiskSafe(e)
\land
SupportStable(e).
$$

其中：

$$
ValuePositive(e)=
\mathbb{1}[S_{\text{value}}(e)\geq\tau_v],
$$

$$
ControlResistant(e)=
\mathbb{1}[S_{\text{control}}(e)\geq\tau_c],
$$

$$
RiskSafe(e)=
\mathbb{1}[P_{\text{bad}}(e)\leq\rho],
$$

$$
SupportStable(e)=
\mathbb{1}[Rel_{\text{support}}(family(e))\geq r_0]
\cdot
\mathbb{1}[Density(e)\geq d_0].
$$

Calibration objective：

$$
\max Coverage(A)
$$

subject to：

$$
Precision_{\text{safe-good}}(A)\geq0.75,
$$

$$
BadEventRate(A)\leq0.05,
$$

$$
Coverage(A)\in[0.03,0.15],
$$

$$
max\_family\_share(A)\leq0.60.
$$

To avoid tiny-slice optimism:

$$
Precision_{\text{LCB}}(A)\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}(A)\leq0.05.
$$

---

# Part V. Candidate statistics

## 1. Value-positive / control-resistant statistics

### VAL0：current exact gap reference

Reference only.

### VAL1：RealGainLCB

Estimate lower confidence of RealFunctional gain within event family:

$$
LCB_{\text{RealGain}}(f)
=
\mu_{Gain_F}(f)
-
\kappa\sqrt{
\frac{\sigma^2_{Gain_F}(f)}{n_f+\epsilon}
}.
$$

Value gate:

$$
S_{\text{value}}(e)=LCB_{\text{RealGain}}(family(e)).
$$

### VAL2：ControlGapLCB

$$
Gap(e)
=
Gain_F(e)-\max(Gain_{\text{AdamWParallel}}(e),Gain_{\text{bestLR}}(e)).
$$

Family lower confidence:

$$
LCB_{\text{Gap}}(f)
=
\mu_{Gap}(f)
-
\kappa\sqrt{
\frac{\sigma^2_{Gap}(f)}{n_f+\epsilon}
}.
$$

### VAL3：NoRegretControlMargin

Avoid functional events whose advantage is only due to weak controls:

$$
S_{\text{noregret}}
=
Gain_F
-
\max(
Gain_{\text{AdamWParallel}},
Gain_{\text{bestLR}},
Gain_{\text{LRScaled}},
Gain_{\text{NoOp}}
).
$$

### VAL4：ValueConsistencyAcrossHorizons

Functional event should remain useful across short and medium horizons:

$$
S_{\text{horizon-consistency}}
=
\min_{h\in\{20,80,240\}}
Gap_h(e).
$$

If only one horizon is available at commit-time, use train-stream proxy accumulated over recent windows, not validation/test.

### VAL5：DeltaGainEfficiency

Reject large movement with low useful gain:

$$
S_{\text{eff}}
=
\frac{Gain_F-\max(Gain_{\text{controls}})}
{\|\Delta z_F\|+\epsilon}.
$$

### VAL6：HybridValueControlLCB

Monotone combination:

$$
S_{\text{value-control}}
=
a_1LCB_{\text{RealGain}}
+
a_2LCB_{\text{Gap}}
+
a_3S_{\text{noregret}}
+
a_4S_{\text{horizon-consistency}}
+
a_5S_{\text{eff}}.
$$

Coefficients chosen on calibration split only, no dataset name.

---

## 2. Risk statistics

### RISK0：v9.2.60 RB3 reference

Reference only.

### RISK1：TailFirstSoftBudgetV2

Tail risk remains hard-ish; other modes use soft budget:

$$
RiskSafe(e)=
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

### RISK2：HarmlessNullAwareRisk

Risk should not accept harmless-null rows just because they are safe. Define:

$$
P_{\text{null}}(e)=\Pr(Y_{\text{harmless-null}}=1\mid x_e).
$$

Then:

$$
RiskUsefulSafe(e)
=
\mathbb{1}[P_{\text{bad}}(e)\leq\rho_b]
\land
\mathbb{1}[P_{\text{null}}(e)\leq\rho_n].
$$

### RISK3：TailVolatilityWindow

Tail instability is computed over train-stream windows:

$$
S_{\text{tail-vol}}
=
CEp99
+
WrongConfP95
-
MarginP10
+
Volatility(CEp99,MarginP10).
$$

### RISK4：ControlDominanceRiskV2

$$
S_{\text{control-risk}}
=
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
-
Gain_F
+
\lambda TailRisk.
$$

### RISK5：RiskValueJointBudget

Accept only if bad risk low and value LCB high:

$$
Budget(e)
=
\lambda_b P_{\text{bad}}(e)
+
\lambda_n P_{\text{null}}(e)
-
\lambda_v S_{\text{value-control}}(e).
$$

Gate:

$$
Budget(e)\leq B_0.
$$

---

## 3. Support statistics

### SUP0：v9.2.60 reference

Reference only.

### SUP1：SignalStratumGeneratorV2

Generate event rows from all combinations:

```text
horizon bucket:
  20,80,240,640

risk mode:
  tail, mismatch, control, family, low-risk

value mode:
  high-gap, mid-gap, low-gap, control-dominant

role mode:
  stack, head, mixed

carrier:
  A0,A3,A6

event source:
  natural, balanced-diagnostic
```

Balanced diagnostic rows must be real train-stream rows, not duplicates.

### SUP2：MultiResolutionFamilyV3

Family definition must not include dataset name:

$$
family(e)=
(
stratum,
horizon,
risk\_mode,
value\_mode,
role\_bucket,
carrier,
control\_gap\_bucket,
tail\_bucket,
delta\_efficiency\_bucket
).
$$

Use empirical-Bayes reliability:

$$
Rel_{\text{EB}}(f)
=
\frac{n_f}{n_f+\alpha}\hat{p}_f
+
\frac{\alpha}{n_f+\alpha}\hat{p}_{parent(f)}.
$$

### SUP3：ValueRiskKNNPocket

Feature vector:

```text
value-control score
bad probability
null probability
tail volatility
delta-gain efficiency
support density
family reliability
horizon bucket
role bucket
```

Density:

$$
Density(e)
=
\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### SUP4：LeaveStratumFamilyOutReliability

Reliability must survive both leave-family-out and leave-stratum-out within calibration:

$$
Rel_{\text{LSFO}}(f)
=
\min(
Rel(f),
\min_{f'\in \mathcal{N}(f)}Rel(f'),
\min_{s'\in \mathcal{N}(s)}Rel(s')
).
$$

### SUP5：BalancedCoverageCap

Global accepted set must satisfy:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
max_stratum_share <= 0.70
```

---

## 4. Controller candidates

### C0：v9.2.60 best reference

Reference only.

### C1：RB3PlusValueGate

Start from RB3 safe region and add value-control gate:

$$
Accept(e)
=
RB3(e)
\land
S_{\text{value-control}}(e)\geq\tau_v
\land
SupportStable(e).
$$

### C2：ValueFirstRiskSecond

$$
Accept(e)
=
S_{\text{value-control}}(e)\geq\tau_v
\land
P_{\text{bad}}(e)\leq\rho_b
\land
P_{\text{null}}(e)\leq\rho_n
\land
SupportStable(e).
$$

### C3：ThreeClassTriageController

Explicit triage:

```text
accept if useful-safe
abstain if harmless-null
reject if bad-event
```

Formula:

$$
Accept(e)
=
\mathbb{1}
[
P_{\text{safe-good}}(e)
-
\max(P_{\text{null}}(e),P_{\text{bad}}(e))
\geq \tau
].
$$

### C4：SupportStratumBalancedController

$$
Accept(e)
=
ValuePositive(e)
\land
ControlResistant(e)
\land
RiskSafe(e)
\land
Rel_{\text{LSFO}}(family(e))\geq r_0
\land
BalancedCoverageCap(e).
$$

### C5：ConstrainedFrontierOptimizerV2

Search over thresholds to maximize coverage:

$$
\max Coverage(A)
$$

subject to:

$$
Precision_{\text{LCB}}(A)\geq0.75,
$$

$$
BadEvent_{\text{UCB}}(A)\leq0.05,
$$

$$
Coverage(A)\in[0.03,0.15],
$$

$$
max\_family\_share\leq0.60.
$$

### C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.60 boundary reproduction

### 目标

确认 v9.2.60 的 boundary 稳定，尤其是 clean core not expandable。

### 必须记录

```text
route
source_route_v9260
clean_core_stable
core_precision
core_coverage
core_bad_event
nearcore_expandable
nearcore_oracle_overlap
risk_budget_best
risk_budget_precision
risk_budget_coverage
risk_budget_bad_event
support_frontier_best
support_frontier_precision
support_frontier_coverage
support_frontier_bad_event
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_bad_event_ucb
true_delta_compute_pass
true_delta_auc
true_delta_agreement
true_delta_step_ratio
oracle_support_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R12-CleanCoreNotExpandable
clean_core_stable = 0
nearcore_expandable = 0
oracle_support_pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_clean_core_failure_ladder.svg
p0_risk_value_support_tradeoff.svg
```

---

## P1：three-class accepted-row decomposition

### 目标

解释 accepted rows 为什么失败：是 bad-event、harmless-null、还是 missing value。

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
safe_good
bad_event
useful
harmless_null
value_positive
control_resistant
tail_safe
support_stable
gap_score
value_score
control_score
bad_probability
null_probability
risk_score
support_score
failure_mode
```

Failure modes：

```text
A1-bad_tail_instability
A2-bad_delta_gain_mismatch
A3-bad_control_dominance
A4-bad_family_unstable
N1-null_control_equivalent
N2-null_low_real_gain
N3-null_low_gap_high_safety
N4-null_support_only
V1-value_missed_by_threshold
V2-control_gap_missed
S1-support_stratum_scarce
```

### 判断标准

P1 pass：

```text
accepted_row_decomposition_fraction >= 0.90
bad_event_mode_attribution_fraction >= 0.90
harmless_null_attribution_fraction >= 0.80
value_miss_attribution_fraction >= 0.80
```

### 可视化

```text
p1_three_class_sankey.svg
p1_bad_vs_null_vs_safe_good.svg
p1_rb3_accepted_decomposition.svg
p1_value_gap_vs_null_scatter.svg
```

---

## P2：support-stratum generator reset

### 目标

真正扩大 balanced diagnostic rows 与 signal strata。P2 是资格门，不是附属诊断。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
risk_modes = tail,mismatch,control,family,low-risk
value_modes = high-gap,mid-gap,low-gap,control-dominant
role_modes = stack,head,mixed
carriers = A0,A3,A6
row_sources = natural,balanced_diagnostic
```

Balanced diagnostic rows 必须是真实 train-stream event rows。禁止复制自然 rows 冒充 balanced rows。

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
risk_mode
value_mode
role_mode
carrier_id
signal_stratum
event_family
source_hash
duplicate_row_flag
safe_good
bad_event
useful
harmless_null
oracle_accept
controller_accept
feature_values
```

### 判断标准

Support generator pass：

```text
natural_real_event_count >= 24000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 32
duplicate_row_count = 0
```

If P2 fails, all downstream official passes are blocked.

### 可视化

```text
p2_signal_strata_coverage.svg
p2_balanced_vs_natural_distribution.svg
p2_family_count_by_stratum.svg
p2_event_generator_coverage_matrix.svg
```

---

## P3：value-positive / control-resistant statistics factory

### 目标

专门解决 RB3 的 low precision：区分 useful safe-good 和 harmless-null。

### 必须记录

```text
value_stat_id
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_useful
AUC_value_positive
AUC_control_resistant
corr_useful
precision_after_value_gate
coverage_after_value_gate
bad_event_after_value_gate
null_rate_after_value_gate
feature_overhead
memory_overhead
```

### 判断标准

Value statistic pass：

$$
AUC_{\text{useful}}\geq0.70
$$

or:

$$
Corr_{\text{useful}}\geq0.35.
$$

Controller utility pass inside RB3 region：

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
AUC_{\text{useful}}\geq0.60
$$

and:

$$
Precision_{\text{RB3+value}} - Precision_{\text{RB3}} \geq0.30.
$$

### 可视化

```text
p3_value_stat_auc_matrix.svg
p3_rb3_plus_value_precision_coverage_bad.svg
p3_value_null_bad_threeway_roc.svg
p3_value_feature_ablation.svg
```

---

## P4：risk-safe / harmless-null aware risk factory

### 目标

风险不再只判 bad-event，还要避免 harmless-null 大量进入 accepted region。

### 必须记录

```text
risk_stat_id
features_used
P_bad_auc
P_null_auc
P_safe_good_auc
bad_event_after_gate
null_rate_after_gate
precision_after_gate
coverage_after_gate
bad_event_delta
null_delta
coverage_delta
```

### 判断标准

Risk-null statistic pass：

$$
AUC_{\text{bad}}\geq0.70,
$$

$$
AUC_{\text{null}}\geq0.65.
$$

Utility pass：

$$
BadEventRate\leq0.05,
$$

$$
Precision_{\text{safe-good}}\geq0.75,
$$

$$
Coverage\geq0.03.
$$

Diagnostic pass：

$$
BadEventRate\leq0.08
$$

and:

$$
Precision_{\text{safe-good}}\geq0.50
$$

with:

$$
Coverage\geq0.03.
$$

### 可视化

```text
p4_bad_null_risk_frontier.svg
p4_harmless_null_rejection_curve.svg
p4_bad_event_reduction_curve.svg
```

---

## P5：support-stratum frontier v2

### 目标

在 P2 expanded support 上重做 support frontier，解决 single-stratum high-precision pocket 问题。

### 必须记录

```text
support_stat_id
family_definition
family_resolution
density_method
reliability_method
leave_stratum_family_out_method
precision
coverage
bad_event
null_rate
legal_oracle_jaccard
oracle_overlap
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
coverage_expansion_factor
```

### 判断标准

Support frontier pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Support balance：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
max_stratum_share <= 0.70
```

Diagnostic support pass：

$$
Jaccard_{\text{legal,oracle}}\geq0.60
$$

or:

$$
Coverage\geq0.01
$$

with:

$$
BadEventRate\leq0.08
$$

and:

$$
accepted_signal_strata_count >= 2.
$$

### 可视化

```text
p5_support_frontier_v2_surface.svg
p5_multistratum_family_heatmap.svg
p5_oracle_legal_overlap_v2.svg
p5_family_reliability_lfo_curve.svg
```

---

## P6：exact reference deployable frontier v5

### 目标

组合 P3/P4/P5，在 CBD0 exact reference 上判断是否存在 deployable accept frontier。

### 必须记录

```text
controller_id
value_stat_id
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
BadEventRate_{\text{heldout}}\leq0.05.
$$

Confidence gate：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
max_stratum_share <= 0.70
```

If no controller passes, route must be:

```text
R10-ExactReferenceStillNotDeployable
```

### 可视化

```text
p6_reference_frontier_v5_precision_coverage_bad.svg
p6_three_class_frontier.svg
p6_value_risk_support_threshold_surface.svg
p6_oracle_legal_gap_after_generator_reset.svg
```

---

## P7：true-delta compute v6 parallel lane

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
risk_support_component_time
value_component_time
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
p7_true_delta_v6_cost_signal_pareto.svg
p7_compute_vs_decision_ladder.svg
p7_residual_subphase_after_value_risk_split.svg
```

---

## P8：system-legal exact-signal controller

### 目标

只有 P6 reference deployable pass 与 P7 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
value_stat_id
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
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p8_system_controller_precision_coverage_bad.svg
p8_system_controller_cost_vs_value.svg
p8_system_controller_family_coverage.svg
```

---

## P9：Leave-dataset-out / leave-stratum-out

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
value_stat_id
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
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_hidden_dataset_tuning_audit.svg
p9_leaveout_failure_modes.svg
```

---

## P10：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P9 survivor
custom_delta_id = best P9 survivor
value_stat_id = best P9 survivor
risk_stat_id = best P9 survivor
support_stat_id = best P9 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledValueStat, ShuffledRiskStat,
           ShuffledSupportStat, ShuffledControlGain, ShuffledCandidateGate,
           ShuffledBranchRatio, ShuffledSignalChannel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
value_stat_id
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
ShuffledValueStat = fail
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
p10_official_paired_replay_pareto.svg
p10_macro_beat_rate.svg
p10_signal_stratum_win_matrix.svg
p10_shuffle_control_matrix.svg
p10_system_gate_distribution.svg
```

---

## P11：Short-run scout

### 目标

如果 P10 pass，验证局部 paired replay 优势能否在连续训练中保持。

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

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9261.csv
p0_v9260_boundary_reproduction.csv
p1_three_class_accepted_row_decomposition.csv
p2_support_stratum_generator_reset.csv
p3_value_control_statistics_factory.csv
p4_risk_null_aware_statistics_factory.csv
p5_support_stratum_frontier_v2.csv
p6_exact_reference_deployable_frontier_v5.csv
p7_true_delta_compute_v6_parallel_lane.csv
p8_system_legal_exact_signal_controller.csv
p9_leave_dataset_and_stratum_out.csv
p10_official_paired_replay.csv
p11_short_run_functional_validation.csv
three_class_trace_v9261.csv
value_control_trace_v9261.csv
risk_null_trace_v9261.csv
support_stratum_generator_trace_v9261.csv
support_frontier_v2_trace_v9261.csv
reference_deployable_frontier_v5_trace.csv
true_delta_compute_v6_trace.csv
system_controller_trace_v9261.csv
leaveout_trace_v9261.csv
paired_replay_branch_trace_v9261.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9260_boundary_unstable
F3_dataset_tuning_detected
F4_three_class_decomposition_incomplete
F5_bad_event_mode_unattributed
F6_harmless_null_unattributed
F7_support_stratum_generator_failed
F8_value_stat_not_predictive
F9_value_gate_precision_fail
F10_risk_null_stat_fail
F11_support_frontier_single_stratum
F12_exact_reference_still_not_deployable
F13_true_delta_compute_still_expensive
F14_true_delta_signal_lost
F15_system_controller_precision_fail
F16_system_controller_coverage_fail
F17_system_controller_bad_event_fail
F18_leave_dataset_out_fail
F19_leave_stratum_out_fail
F20_paired_replay_control_equivalent
F21_shuffle_control_pass
F22_functional_lr_equivalent
F23_short_run_task_drop
F24_full_run_no_macro_hard_stratum_gain
F25_strong_baseline_explains_gain
F26_robustness_fail
F27_external_not_ready
F28_fake_or_proxy_violation
F29_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.60 boundary is reproduced.

R2-ThreeClassDecompositionPass:
  accepted rows are decomposed into safe-good / harmless-null / bad-event.

R3-SupportStratumGeneratorPass:
  support generator reaches >=6 signal strata and >=6000 balanced diagnostic rows.

R4-ValueControlStatisticPass:
  legal value/control statistic separates useful events from harmless-null.

R5-RiskNullStatisticPass:
  bad-event and harmless-null risk are jointly controlled.

R6-SupportFrontierV2Pass:
  support frontier reaches multi-stratum / multi-family balanced coverage.

R7-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + value/risk/support frontier passes precision / coverage / bad-event gate.

R8-TrueDeltaComputeV6SystemPass:
  true branch-delta v6 passes system envelope while preserving signal.

R9-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R10-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R11-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R12-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R13-ValueRiskOrthogonalizationFail:
  risk-safe region exists but useful/value precision cannot be recovered.

R14-SupportStratumGeneratorFail:
  support remains narrow or single-stratum after generator reset.

R15-ExactReferenceStillNotDeployable:
  exact reference still cannot form deployable accept frontier.

R16-ReferenceFeasibleButComputeFail:
  decision geometry viable, true-delta compute still too expensive.

R17-ComputePassButReferenceFail:
  compute viable, accept-region geometry still fails.

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
v9260_boundary_pass
dataset_tuning_detected
three_class_decomposition_pass
bad_event_attribution_fraction
harmless_null_attribution_fraction
value_miss_attribution_fraction
support_stratum_generator_pass
natural_real_event_count
balanced_diagnostic_real_event_count
measured_signal_strata_count
measured_family_count
duplicate_row_count
best_value_stat_id
value_stat_pass
value_auc_useful
value_precision_after_gate
value_coverage_after_gate
best_risk_null_stat_id
risk_null_stat_pass
bad_auc
null_auc
bad_event_after_gate
null_rate_after_gate
best_support_frontier_id
support_frontier_pass
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
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
success_v9261_strict_purekan_functional
success_v9261_full_functional
success_v9261_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 three-class accepted-row decomposition
  P2 support-stratum generator reset
  P3 value-control statistics factory
  P4 risk-null aware statistics factory
  P5 support-stratum frontier v2
  P6 exact reference deployable frontier v5
  P7 true-delta compute v6

Batch 2:
  P8 system-legal controller calibration
  P9 leave-dataset-out / leave-stratum-out
  P10 paired replay scout

Batch 3:
  official P10 paired replay
  P11 short-run if paired replay passes

Batch 4:
  full 10-seed / robustness / strong baseline only if P11 passes
```

Gate rule：

```text
P6 exact reference deployable frontier can pass diagnostically before compute pass.
P8 official controller cannot pass unless P6 reference deployable frontier and P7 compute pass.
P9/P10 diagnostic rows may be measured before all gates finish.
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

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.60 boundary reproduced
accepted rows decomposed into safe-good / harmless-null / bad-event
support-stratum generator reset measured
value-control statistics measured
risk-null statistics measured
support frontier v2 measured
exact reference deployable frontier measured
true-delta compute v6 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
support-stratum generator pass
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
1. v9.2.60 boundary cannot be reproduced；
2. three-class decomposition cannot attribute accepted-row failures；
3. support-stratum generator cannot exceed 3 signal strata；
4. support measurement remains too narrow；
5. value-control statistic cannot separate useful from harmless-null；
6. risk-null statistic cannot reduce bad and null simultaneously；
7. exact reference remains not deployable；
8. true branch-delta compute remains too expensive；
9. true branch-delta compute passes system but loses signal；
10. system-legal controller cannot meet precision / coverage / bad-event；
11. leave-dataset-out fails；
12. leave-stratum-out fails；
13. paired replay remains control-equivalent；
14. shuffle controls pass, indicating overfit；
15. short-run task drops；
16. full run gives no macro / hard-stratum / geometry gain；
17. functional breaks system gate；
18. gains are explained by QuadraticFeatureMLP；
19. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：value-risk-support frontier + compute + LDO/LSO + paired replay pass

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

下一步继续 kernel / layout / fused value-risk-support compute，不调 dataset。

## Case C：compute pass but reference frontier fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 value-risk-support sufficient statistics，不走 kernel-only。

## Case D：value-risk orthogonalization fail

必须声明：

```text
risk-safe region exists, but legal features cannot recover useful/value precision.
```

下一步重设 value/control target decomposition and output-delta sufficient statistics。

## Case E：support-stratum generator fail

必须声明：

```text
controller evidence remains narrow-support; no official promotion.
```

下一步先修 event generation and signal-stratum coverage。

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

# Part XII. 最终建议

v9.2.61 的一句话策略是：

$$
\boxed{
\text{不要再扩张一个不稳定 clean core；先把 accepted rows 分成 useful / null / bad，再用 value-risk-support 正交 frontier 做 deployable controller。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
TBD0 有没有 AUC；
K7d local fusion 是否能跑；
formula proxy 是否够快；
clean core 能不能再微调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. RB3 的 low precision 到底是哪些 harmless-null rows 造成的？
2. useful/value-positive/control-resistant 是否有 legal online signature？
3. bad-event 与 harmless-null 能否同时被建模？
4. signal strata 为什么始终只有 3？
5. support generator 能否真实产生 >=6 strata / >=6000 balanced rows？
6. exact reference deployable frontier 能否达到 coverage >=0.03？
7. true delta compute v6 能否继续压到 step <=1.50？
8. system-legal controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.61 的结果将给出清晰分叉：

```text
if value-risk-support frontier restores reference deployability and true-delta compute passes:
  open system-legal controller, LDO/LSO, paired replay.

if reference deployable but compute fail:
  kernelization remains blocker.

if compute pass but reference frontier fail:
  value-risk-support decision geometry remains blocker.

if support generator cannot expand strata:
  no official promotion; reset event generation.

if value-control features cannot distinguish useful from harmless-null:
  exact branch-delta remains diagnostic; reset value/control sufficient statistics.

if oracle support collapses:
  carrier/support stability is blocker.
```
