# DG-KAN v9.2.65 Joint Safe-Useful Feasibility 与 Null-Bad Conflict Resolution 完整实验计划

> 本计划基于 v9.2.64 `Conditional Null Separation 与 Support-Family Densification` 的真实复盘制定。  
> v9.2.64 的 terminal route 是：
>
> ```text
> route = R13-SafeUsefulStillTiny
> base_candidate = LQ-t2-h256
> success_v9264_strict_purekan_functional = False
> success_v9264_full_functional = False
> success_v9264_external_ready = False
> ```
>
> v9.2.64 的关键事实是：
>
> ```text
> P1:
>   conditional-null autopsy pass = 1
>   SU2 count = 6527
>   harmless-null count = 7955
>   bad-event count = 2666
>   safe-useful miss count = 982
>   attribution fractions = 1.0
>
> P2:
>   best conditional-null statistic = CN6-FamilyNullUCB
>   conditional-null AUC = 0.893849
>   corr = 0.714209
>   null rate after gate = 0.020472
>   precision = 0.472441
>   bad-event = 0.374803
>   utility / diagnostic pass = 0
>
> P3:
>   support-family densification pass = 1
>   natural rows = 24192
>   balanced diagnostic rows = 6000
>   measured signal strata = 318
>   measured family count = 2730
>   duplicate rows = 0
>
> P4:
>   best null-aware safe-useful score = NASU4-UsefulPocketScore
>   safe-useful AUC = 0.960014
>   coverage = 0.069775
>   null rate = 0.064771
>   precision = 0.353870
>   bad-event = 0.451817
>   precision LCB = 0.317603
>
> P5:
>   best exact-reference controller =
>     C-NASU2-SupportBackedValue
>     + CN7-ConditionalNullMixture
>     + CB6-ConditionalBadMixture
>     + SF4-NullAwareSupportPocket
>
>   precision = 0.813953
>   bad-event = 0.046512
>   null rate = 0.139535
>   coverage = 0.004740
>   precision LCB = 0.673827
>   bad-event UCB = 0.154558
>   exact_reference_deployable = 0
>
> P6:
>   true-delta compute pass = 0
>   true_delta_step_ratio_q90 = 2.863280 > 1.50
> ```
>
> v9.2.64 的核心判断是：
>
> $$
> \boxed{
> \text{null 和 support 已经有强信号，但 safe-useful 交集仍不能形成 deployable frontier。}
> }
> $$
>
> 这不是 functional update 失败，也不是 basis 或 carrier 退化。更准确地说，v9.2.64 证明了：
>
> $$
> \boxed{
> \text{conditional-null 可预测，support-family 足够宽，conditional-bad 也有信号。}
> }
> $$
>
> 但它也证明：
>
> $$
> \boxed{
> \text{把这些信号用顺序 gate 串起来，会在 “unsafe broad region” 和 “safe tiny region” 之间震荡。}
> }
> $$
>
> 因此 v9.2.65 不应继续只做：
>
> ```text
> 再加一个 null score；
> 再硬化 bad-event gate；
> 再扩 support rows；
> 再调 NASU threshold；
> 再做 kernel-only；
> 按 dataset 单独调 controller。
> ```
>
> 本轮的核心问题必须变成：
>
> $$
> \boxed{
> \text{在当前 labels 和 event distribution 下，safe-useful deployable region 是否存在？}
> }
> $$
>
> 如果 oracle-bounded safe-useful frontier 不存在，继续设计 legal controller 没意义；必须重设 target / event definition。  
> 如果 oracle-bounded frontier 存在，但 legal scores 找不到，说明是 representation/controller 问题。  
> 如果 legal exact-reference frontier 过了而 true-delta compute 不过，才回到 kernelization blocker。  
> v9.2.65 要把这三个分支明确分开。

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
  report safe-useful / risky-useful / harmless-null / bad-null by dataset
  report null-bad conflict by dataset
  report family reliability by dataset
  report leave-dataset-out failure matrix
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: threshold = A
  if dataset == KMNIST: use controller B
  if dataset == MNIST: skip borderline rows
  tune thresholds per dataset
  choose support family per dataset
  choose controller per dataset
  choose selected/full-logit mode per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-null-support-family-level / branch-delta features。Dataset name 只能进入 report，不能进入 commit rule。

---

# Part I. 对 v9.2.64 的独立判断

## 1. v9.2.64 没有达到目标

v9.2.64 没有 strict PureKAN functional success。失败不是 paired replay 输了，而是 paired replay 根本没有资格打开：

```text
exact_reference_deployable = 0
true_delta_compute_pass = 0
system_legal_controller = not_run
LDO / LSO / paired replay = not_run
```

P5 的 best exact-reference controller 有一组看似不错的 point estimates：

```text
precision = 0.813953
bad-event = 0.046512
null rate = 0.139535
```

但它不能 official，因为：

```text
coverage = 0.004740 < 0.03
precision LCB = 0.673827 < 0.75
bad-event UCB = 0.154558 > 0.05
```

这不是 near-pass，而是 **small clean-ish pocket**。它距离 deployable frontier 至少还差一个数量级的 coverage，同时置信边界不过。

## 2. v9.2.64 的真实进展

v9.2.64 有三项关键推进。

第一，support-family densification 成功。v9.2.63 的 family count 只有 `538 < 700`，v9.2.64 提升到：

```text
measured signal strata = 318
measured family count = 2730
duplicate rows = 0
```

这说明 support-family regression 被修复，当前不能再把主要失败归因于 family count 不够。

第二，conditional-null statistics 有强信号。`CN6-FamilyNullUCB` 的 AUC 达到 `0.893849`，corr 达到 `0.714209`，并且能把 null rate 压到 `0.020472`。这说明 harmless-null 不是不可预测。

第三，null-aware safe-useful score 的 ranking signal 很强。`NASU4-UsefulPocketScore` 的 AUC 达到 `0.960014`，coverage `0.069775`，null rate `0.064771`，说明它确实能找到 broad low-null candidate region。

## 3. v9.2.64 的真实失败

v9.2.64 的失败不是单一 feature 不强，而是 **强 signal 之间的联合几何不闭合**。

P2 的 null gate 后：

```text
null rate = 0.020472
precision = 0.472441
bad-event = 0.374803
```

说明 null gate 很强，但它把 bad-event 留下来了。

P4 的 best score：

```text
AUC = 0.960014
coverage = 0.069775
null rate = 0.064771
precision = 0.353870
bad-event = 0.451817
```

说明 broad region 够宽、null 也够低，但 safe-useful precision 很差，bad-event 很高。

P5 的 exact-reference frontier：

```text
precision = 0.813953
bad-event = 0.046512
null = 0.139535
coverage = 0.004740
```

说明如果强行同时压 precision/bad/null，就退化成 tiny slice。

这三个结果合起来说明：

$$
\boxed{
\text{当前不是 “缺一个更强 null score”，而是 safe-useful 的 joint feasible region 没有被证明存在。}
}
$$

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{safe-useful joint feasibility unknown / likely too narrow under current target decomposition。}
}
$$

更具体地说：

```text
1. exact branch-delta signal 仍强；
2. conditional bad-event 有可测信号；
3. conditional null 有强信号；
4. support-family 已经足够宽；
5. broad low-null / low-bad-looking region 仍可能 unsafe；
6. strict safe-useful intersection tiny；
7. true-delta compute 仍太贵。
```

所以 v9.2.65 必须先做 **oracle-bounded feasibility audit**。如果 oracle 在当前 event set / target definition 下都无法提供 coverage >= 0.03 的 safe-useful accept set，那么所有 legal controller 都不可能成功。反过来，如果 oracle 能做到，但 legal scores 做不到，才说明要继续修 sufficient statistics。

---

# Part II. v9.2.65 总体目标

v9.2.65 的总体目标是：

$$
\boxed{
\text{判定 safe-useful deployable frontier 是否存在，并把 null/bad/value/support 从 sequential gates 改成 joint feasibility model。}
}
$$

本轮必须并行回答六个问题：

```text
Q1:
  在当前 labels / rows / families 中，oracle-bounded safe-useful frontier 是否存在？

Q2:
  如果 oracle frontier 存在，legal features 为什么无法达到同样 coverage？

Q3:
  null gate 与 bad-event gate 是否冲突，即 low-null rows 是否系统性 high-bad？

Q4:
  broad NASU-like region 中的 false-positive 到底是 bad、null、control-equivalent 还是 target ambiguity？

Q5:
  exact-reference frontier 能否通过 joint 4-class model 进入 coverage >= 0.03？

Q6:
  如果 decision geometry 恢复，true-delta compute 是否仍是唯一 blocker？
```

v9.2.65 的核心 stop-go 不是 “AUC 是否更高”，而是：

$$
\boxed{
\text{coverage-preserving safe-useful frontier 是否可存在、可观测、可部署。}
}
$$

---

# Part III. 核心假设

## H1：当前 exact-reference failure 可能是 target-support infeasibility，而不是 controller weakness

v9.2.64 P5 best coverage 只有 `0.004740`。H1 认为需要先测当前 target 的 oracle upper bound。

H1 成立标准：

Oracle-bounded frontier 能达到：

$$
Precision_{\text{oracle}}\geq0.75,
$$

$$
Coverage_{\text{oracle}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{oracle}}\leq0.05,
$$

$$
NullRate_{\text{oracle}}\leq0.15.
$$

H1 失败标准：

即使 oracle 使用 posthoc safe-useful labels，也只能得到：

$$
Coverage_{\text{oracle}}<0.03
$$

或需要牺牲：

$$
BadEventRate>0.05
$$

才能达到 coverage。若 H1 失败，下一步必须重设 safe-useful target 或 event generator，而不是继续调 legal controller。

## H2：null/bad gates 存在冲突；单独强的 gates 串联后会坍缩

证据：

```text
CN6 gate:
  null rate low, but bad-event high.

NASU4:
  AUC high, coverage high, null low, but bad-event high.

P5:
  bad/null/precision point estimates pass, but coverage tiny.
```

H2 成立标准：

在 joint matrix 中观察到：

$$
P(Bad=1\mid NullLow) - P(Bad=1) \geq 0.10
$$

或：

$$
Corr(S_{\text{null}},S_{\text{bad}})<-0.30
$$

with safety conflict in accepted candidate region.

H2 失败标准：

null/bad gates 并不冲突；失败来自 value/control signal 或 calibration only。

## H3：broad NASU-like region 的主要错误不是 null，而是 risky-useful / bad-null 混合

v9.2.64 P4 null rate 已经只有 `0.064771`，但 precision 只有 `0.353870`，bad-event `0.451817`。H3 认为该 region 的 false positives 主要是 bad-related，而不是 harmless-null。

H3 成立标准：

在 NASU4 accepted region 中：

```text
bad-related false positives >= 60% of all false positives
harmless-null-only false positives <= 25%
```

H3 失败标准：

false positives 仍以 harmless-null 为主，说明 conditional-null 仍未真正解决。

## H4：safe-useful score 需要直接建模 joint class，而不是 sequential value / bad / null gates

定义四类：

```text
SU = safe-useful
RU = risky-useful
HN = harmless-null
BN = bad-null
```

H4 成立标准：

Joint 4-class controller 达到：

$$
AUC_{\text{SU vs rest}}\geq0.70
$$

or:

$$
Corr(S_{\text{joint}},Y_{\text{SU}})\geq0.35
$$

并且 heldout pass：

$$
Precision\geq0.75,\quad
Coverage\in[0.03,0.15],\quad
BadEvent\leq0.05,\quad
NullRate\leq0.15.
$$

H4 失败标准：

joint score 仍只能在 coverage `<0.01` 下安全。

## H5：true-delta compute 仍是并行 blocker，但本轮主 blocker 是 decision feasibility

H5 成立标准：

至少一个 true-delta candidate 达到：

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

true-delta compute 仍：

$$
StepRatio_{q90}>1.50
$$

or signal lost。

如果 H1-H4 失败，即使 H5 成立也不能进入 functional success，因为 exact-reference decision geometry 不可部署。

---

# Part IV. 新目标分解：从 sequential gate 到 joint feasibility

## 1. 四类 joint label

每个 event 被分成四类：

$$
Y_{\text{SU}}=
Y_{\text{useful}}
\land
(1-Y_{\text{bad}})
\land
(1-Y_{\text{null}}).
$$

$$
Y_{\text{RU}}=
Y_{\text{useful}}
\land
Y_{\text{bad}}.
$$

$$
Y_{\text{HN}}=
(1-Y_{\text{useful}})
\land
(1-Y_{\text{bad}})
\land
Y_{\text{null}}.
$$

$$
Y_{\text{BN}}=
(1-Y_{\text{useful}})
\land
Y_{\text{bad}}.
$$

如果某个 row 同时满足多个 weak labels，则必须记录 conflict flag：

```text
label_conflict_useful_bad
label_conflict_null_bad
label_conflict_support_only
label_conflict_horizon
```

这些 conflict rows 不允许被静默合并到 safe-useful。

## 2. Oracle-bounded frontier

定义 oracle accept set：

$$
A_{\text{oracle}}(\tau)
=
\{e: Y_{\text{SU}}(e)=1\}.
$$

但为了模拟 official controller 的 coverage/balance，还要加 support constraints：

$$
accepted\_signal\_strata\_count(A)\geq5,
$$

$$
accepted\_family\_count(A)\geq32,
$$

$$
max\_family\_share(A)\leq0.50.
$$

Oracle feasibility 不是 official success；它是判断当前 target 是否可行的上界。

## 3. Joint safe-useful score

旧模型：

$$
Accept
=
Useful
\land
RiskSafe
\land
NullRejected
\land
SupportStable.
$$

新模型：

$$
S_{\text{joint}}(e)
=
LCB[P(SU\mid x_e)]
-
\lambda_1 UCB[P(RU\mid x_e)]
-
\lambda_2 UCB[P(HN\mid x_e)]
-
\lambda_3 UCB[P(BN\mid x_e)]
+
\lambda_4 LCB[Support(e)].
$$

Accept：

$$
Accept(e)
=
\mathbb{1}[S_{\text{joint}}(e)\geq\tau]
\land
\mathbb{1}[UCB(P(RU)+P(BN))\leq\rho_B]
\land
\mathbb{1}[UCB(P(HN))\leq\rho_N]
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

---

# Part V. Candidate statistics

## 1. Oracle / feasibility candidates

### OF0：Unconstrained safe-useful oracle

Posthoc diagnostic only. Measures upper bound:

$$
A_{\text{OF0}}=\{e:Y_{\text{SU}}=1\}.
$$

### OF1：Support-balanced safe-useful oracle

Adds family/stratum balance:

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### OF2：Exact-reference oracle-compatible frontier

Uses CBD0 exact reference signal order but posthoc labels for feasibility sweep. Diagnostic only.

### OF3：Leave-family oracle

Measures whether oracle support survives leave-family-out.

## 2. Null-bad conflict statistics

### NBC0：Reference conflict matrix

Diagnostic only.

### NBC1：Null-low bad-risk UCB

$$
S_{\text{conflict}}
=
\mathbb{1}[UCB_{\text{null}}\leq\rho_N]
\cdot
UCB_{\text{bad}}.
$$

### NBC2：Bad-low null-risk UCB

$$
S_{\text{badlow-null}}
=
\mathbb{1}[UCB_{\text{bad}}\leq\rho_B]
\cdot
UCB_{\text{null}}.
$$

### NBC3：Joint conflict penalty

$$
Penalty_{\text{conflict}}
=
\alpha_1 UCB(RU)
+
\alpha_2 UCB(BN)
+
\alpha_3 UCB(HN)
+
\alpha_4 I_{\text{label-conflict}}.
$$

## 3. Joint class statistics

### JC1：FourClassLCB-UCB

For each family / pocket:

$$
LCB_{\text{SU}}(f)
=
\hat{p}_{SU}(f)
-
\kappa
\sqrt{\frac{\hat{p}_{SU}(f)(1-\hat{p}_{SU}(f))}{n_f+\epsilon}}.
$$

$$
UCB_{\text{RU/HN/BN}}(f)
=
\hat{p}_{class}(f)
+
\kappa
\sqrt{\frac{\hat{p}_{class}(f)(1-\hat{p}_{class}(f))}{n_f+\epsilon}}.
$$

### JC2：JointMarginScore

$$
S_{\text{joint-margin}}
=
LCB_{\text{SU}}
-
\max(
UCB_{\text{RU}},
UCB_{\text{HN}},
UCB_{\text{BN}}
).
$$

### JC3：RiskWeightedUsefulDensity

Feature vector:

```text
safe-useful score
conditional bad UCB
conditional null UCB
control gap LCB
delta efficiency
horizon persistence
family reliability
support density
```

Density:

$$
Density_{\text{joint}}(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

### JC4：HorizonPersistentSafeUseful

Reject useful-looking rows that do not persist across horizons:

$$
S_{\text{HP-SU}}
=
\min_{h\in\{20,80,240,640\}}
[
Gain_{F,h}-\max_c Gain_{c,h}
]
-
\lambda_B UCB_{\text{bad}}
-
\lambda_N UCB_{\text{null}}.
$$

### JC5：HybridJointSafeUseful

$$
S_{\text{hybrid-joint}}
=
a_1S_{\text{joint-margin}}
+
a_2Density_{\text{joint}}
+
a_3S_{\text{HP-SU}}
+
a_4LCB_{\text{support}}
-
a_5Penalty_{\text{conflict}}.
$$

Coefficients are selected on calibration split only. Dataset name is forbidden.

## 4. Controller candidates

### C0：v9.2.64 best reference

Reference only.

### C1：OracleFeasibilityController

Posthoc diagnostic only. Measures whether target exists.

### C2：JointMarginController

$$
Accept(e)
=
S_{\text{joint-margin}}(e)\geq\tau
\land
SupportStable(e).
$$

### C3：ConflictPenalizedController

$$
Accept(e)
=
S_{\text{hybrid-joint}}(e)\geq\tau
\land
Penalty_{\text{conflict}}(e)\leq\pi
\land
SupportStable(e).
$$

### C4：HorizonPersistentSafeUsefulController

$$
Accept(e)
=
S_{\text{HP-SU}}(e)\geq\tau_H
\land
UCB_{\text{bad}}(e)\leq\rho_B
\land
UCB_{\text{null}}(e)\leq\rho_N
\land
SupportStable(e).
$$

### C5：ConstrainedJointOptimizer

Search thresholds to maximize coverage:

$$
\max Coverage(A)
$$

subject to all official constraints.

### C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.64 boundary reproduction

### 目标

确认最新 boundary 稳定，尤其是 R13-SafeUsefulStillTiny、support-family densification pass、P5 exact-reference frontier fail。

### 必须记录

```text
route
source_route_v9263
conditional_null_autopsy_pass
conditional_null_stat_pass
support_family_densification_pass
null_aware_safe_useful_score_pass
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
true_delta_compute_pass
true_delta_step_ratio_q90
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R13-SafeUsefulStillTiny
support_family_densification_pass = 1
exact_reference_deployable = 0
true_delta_compute_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_v9263_to_v9264_progress_ladder.svg
p0_safe_useful_tiny_frontier.svg
```

---

## P1：oracle-bounded safe-useful feasibility audit

### 目标

先判断当前 target / row distribution 是否存在 deployable safe-useful region。  
这是 v9.2.65 的第一硬门。如果 P1 不过，后续 legal controller 全部只能 diagnostic。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
safe_useful
risky_useful
harmless_null
bad_null
bad_event
null_event
support_stable
oracle_accept
oracle_support_balanced_accept
family_id
stratum_id
label_conflict_flags
```

必须输出：

```text
oracle_unconstrained_precision
oracle_unconstrained_coverage
oracle_unconstrained_bad_event
oracle_unconstrained_null_rate

oracle_support_balanced_precision
oracle_support_balanced_coverage
oracle_support_balanced_bad_event
oracle_support_balanced_null_rate

oracle_leave_family_precision
oracle_leave_family_coverage
oracle_leave_family_bad_event
oracle_leave_family_null_rate
```

### 判断标准

Oracle feasibility pass：

$$
Precision_{\text{oracle}}\geq0.75,
$$

$$
Coverage_{\text{oracle}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{oracle}}\leq0.05,
$$

$$
NullRate_{\text{oracle}}\leq0.15.
$$

Support-balanced oracle pass：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p1_oracle_feasibility_frontier.svg
p1_safe_useful_base_rate_by_stratum_family.svg
p1_oracle_support_balance_heatmap.svg
p1_oracle_vs_legal_gap.svg
```

---

## P2：null-bad conflict autopsy

### 目标

判断 null gate 与 bad-event gate 是否在当前 feature space 中冲突。

### 必须记录

```text
row_id
safe_useful
risky_useful
harmless_null
bad_null
CN_score
CB_score
NASU_score
support_score
low_null
low_bad
low_null_high_bad
low_bad_high_null
accepted_by_CN
accepted_by_CB
accepted_by_NASU
accepted_by_P5_best
failure_mode
```

Conditional rates：

```text
P(bad | low_null)
P(null | low_bad)
P(safe_useful | low_null, low_bad)
P(risky_useful | low_null)
P(harmless_null | low_bad)
P(bad_null | low_null)
```

### 判断标准

P2 pass：

```text
conflict_matrix_complete = 1
bad/null conditional rates measured on heldout
all major failure cells attributed fraction >= 0.90
```

Conflict detected if：

$$
P(Bad\mid LowNull)-P(Bad)\geq0.10
$$

or:

$$
P(Null\mid LowBad)-P(Null)\geq0.10.
$$

### 可视化

```text
p2_null_bad_conflict_matrix.svg
p2_cn_vs_cb_scatter.svg
p2_low_null_high_bad_sankey.svg
p2_low_bad_high_null_sankey.svg
```

---

## P3：four-class joint statistic factory

### 目标

从 sequential gate 改为 joint four-class score，直接区分 safe-useful / risky-useful / harmless-null / bad-null。

### 必须记录

```text
joint_stat_id
features_used
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_SU_vs_rest
corr_SU
AUC_RU
AUC_HN
AUC_BN
precision_after_gate
coverage_after_gate
bad_event_after_gate
null_rate_after_gate
precision_lcb
bad_event_ucb
accepted_strata_count
accepted_family_count
```

### 判断标准

Joint statistic pass：

$$
AUC_{\text{SU vs rest}}\geq0.70
$$

or:

$$
Corr(S_{\text{joint}},Y_{\text{SU}})\geq0.35.
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
p3_four_class_auc_matrix.svg
p3_joint_margin_frontier.svg
p3_four_class_confusion.svg
p3_joint_score_ablation.svg
```

---

## P4：family-stable joint support audit

### 目标

support-family 已过，但要确认 safe-useful joint frontier 在 leave-family / leave-stratum 下不退化。

### 必须记录

```text
row_id
family_coarse
family_mid
family_fine
signal_stratum
support_density
family_reliability_SU
family_reliability_RU
family_reliability_HN
family_reliability_BN
leave_family_out_reliability
leave_stratum_out_reliability
duplicate_row_flag
source_hash
```

### 判断标准

Support stability pass：

```text
natural_real_event_count >= 24192
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 300
measured_family_count >= 2000
duplicate_row_count = 0
```

Accepted support pass：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
leave_family_out_safe = 1
leave_stratum_out_safe = 1
```

### 可视化

```text
p4_joint_family_reliability_heatmap.svg
p4_leave_family_out_reliability.svg
p4_leave_stratum_out_reliability.svg
p4_accepted_support_balance.svg
```

---

## P5：exact-reference deployable frontier v9

### 目标

组合 P2/P3/P4，在 CBD0 exact reference 上判断 safe-useful deployable frontier 是否恢复。  
P5 是 v9.2.65 的主要成功门。

### 必须记录

```text
controller_id
joint_stat_id
conflict_stat_id
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
oracle_gap
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
R10-OracleSafeUsefulInfeasible
R11-JointSafeUsefulStatisticFail
R12-ExactReferenceStillTinyAfterJointModel
R13-NullBadConflictUnresolved
```

### 可视化

```text
p5_reference_frontier_v9_precision_coverage_bad_null.svg
p5_oracle_vs_legal_frontier.svg
p5_joint_threshold_surface.svg
p5_deployable_region_ladder.svg
```

---

## P6：true-delta compute v10 parallel lane

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
AUC_joint_SU
AUC_conditional_bad
AUC_conditional_null
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
joint_score_component_time
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
p6_true_delta_v10_cost_signal_pareto.svg
p6_compute_vs_decision_ladder.svg
p6_residual_subphase_after_joint_model.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P5 reference deployable pass 与 P6 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
joint_stat_id
conflict_stat_id
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
joint_stat_id
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
joint_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledJointScore, ShuffledConflictPenalty,
           ShuffledConditionalNull, ShuffledConditionalBad, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledSignalChannel, FunctionalChannelShuffled, TailMaskShuffled,
           RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled,
           InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
joint_stat_id
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
ShuffledJointScore = fail
ShuffledConflictPenalty = fail
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
contract_audit_v9265.csv
p0_v9264_boundary_reproduction.csv
p1_oracle_bounded_safe_useful_feasibility.csv
p2_null_bad_conflict_autopsy.csv
p3_four_class_joint_statistic_factory.csv
p4_family_stable_joint_support_audit.csv
p5_exact_reference_deployable_frontier_v9.csv
p6_true_delta_compute_v10_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
oracle_feasibility_trace_v9265.csv
null_bad_conflict_trace_v9265.csv
four_class_joint_trace_v9265.csv
joint_support_trace_v9265.csv
reference_frontier_v9_trace.csv
true_delta_compute_v10_trace.csv
system_controller_trace_v9265.csv
leaveout_trace_v9265.csv
paired_replay_branch_trace_v9265.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9264_boundary_unstable
F3_dataset_tuning_detected
F4_oracle_safe_useful_infeasible
F5_oracle_support_balance_fail
F6_null_bad_conflict_unattributed
F7_joint_stat_not_predictive
F8_joint_stat_tiny_coverage
F9_precision_lcb_fail
F10_bad_event_ucb_fail
F11_null_rate_fail
F12_support_balance_fail
F13_exact_reference_still_tiny_after_joint_model
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
  v9.2.64 boundary reproduced.

R2-OracleSafeUsefulFeasible:
  current target / rows contain deployable oracle safe-useful frontier.

R3-NullBadConflictAttributed:
  null-bad conflict is measured and attributed.

R4-FourClassJointStatisticPass:
  legal joint statistic separates SU/RU/HN/BN and passes utility gate.

R5-FamilyStableJointSupportPass:
  joint support remains broad and leave-family/leave-stratum stable.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + joint frontier passes precision / coverage / bad-event / null gate.

R7-TrueDeltaComputeV10SystemPass:
  true branch-delta v10 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-OracleSafeUsefulInfeasible:
  current safe-useful target/event distribution has insufficient oracle coverage.

R13-NullBadConflictUnresolved:
  null/bad signals conflict and legal features cannot resolve them.

R14-JointSafeUsefulStillTiny:
  joint score can be safe only at coverage < 0.03.

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
v9264_boundary_pass
dataset_tuning_detected
oracle_safe_useful_feasible
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
oracle_support_balanced_pass
null_bad_conflict_attributed
P_bad_given_low_null
P_null_given_low_bad
best_joint_stat_id
joint_stat_pass
joint_auc_su
joint_precision
joint_coverage
joint_bad_event
joint_null_rate
precision_lcb
bad_event_ucb
support_joint_pass
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
true_delta_safe_useful_auc
true_delta_conditional_bad_auc
true_delta_conditional_null_auc
true_delta_joint_auc
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
oracle_support_precision
oracle_support_coverage
oracle_support_bad_event
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9265_strict_purekan_functional
success_v9265_full_functional
success_v9265_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 oracle-bounded feasibility
  P2 null-bad conflict autopsy
  P3 four-class joint statistic factory
  P4 family-stable joint support audit
  P5 exact reference deployable frontier v9
  P6 true-delta compute v10

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
P1 oracle feasibility can run before all legal controller gates.
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
v9.2.64 boundary reproduced
oracle-bounded feasibility measured
null-bad conflict measured
four-class joint statistic measured
family-stable joint support audited
exact reference deployable frontier v9 measured
true-delta compute v10 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
oracle safe-useful feasible
+
joint statistic pass
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
1. v9.2.64 boundary cannot be reproduced；
2. oracle safe-useful frontier infeasible；
3. null-bad conflict cannot be attributed；
4. joint safe-useful statistic cannot predict SU vs rest；
5. joint score remains tiny coverage；
6. precision LCB remains below 0.75；
7. bad-event UCB remains above 0.05；
8. null rate remains above 0.15；
9. support balance fails；
10. exact reference remains not deployable after joint model；
11. true branch-delta compute remains too expensive；
12. true branch-delta compute passes system but loses signal；
13. system-legal controller cannot meet precision / coverage / bad-event / null-rate；
14. leave-dataset-out fails；
15. leave-stratum-out fails；
16. paired replay remains control-equivalent；
17. shuffle controls pass, indicating overfit；
18. short-run task drops；
19. full run gives no macro / hard-stratum / geometry gain；
20. functional breaks system gate；
21. gains are explained by QuadraticFeatureMLP；
22. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：oracle infeasible

必须声明：

```text
current safe-useful target / event distribution does not contain enough deployable support.
```

下一步应重设 safe-useful target decomposition 或 event generator，不继续调 legal controller。

## Case B：oracle feasible, legal reference fail

必须声明：

```text
target exists, but current legal sufficient statistics cannot recover it.
```

下一步继续修 joint SU/RU/HN/BN representation。

## Case C：reference deployable, compute fail

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused joint-score compute，不调 dataset。

## Case D：compute pass, reference fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 joint safe-useful decision geometry，不走 kernel-only。

## Case E：joint score safe only at tiny coverage

必须声明：

```text
current features can isolate a tiny clean pocket but cannot recover deployable support.
```

下一步要重设 output-delta sufficient statistics or target semantics。

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

v9.2.65 的一句话策略是：

$$
\boxed{
\text{不要再顺序串联 null/bad/value gates；先判断 safe-useful oracle frontier 是否存在，再用四类 joint model 恢复 coverage。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
conditional-null AUC 是否存在；
support family count 是否够；
NASU score 是否有 AUC；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 当前 safe-useful target 在 oracle 上是否有 coverage >=0.03？
2. low-null 是否天然 high-bad？
3. low-bad 是否天然 high-null？
4. NASU4 broad region 的 false positives 到底是 RU、HN 还是 BN？
5. 四类 SU/RU/HN/BN joint model 能否比 sequential gate 更稳定？
6. exact-reference coverage 能否从 0.004740 拉到 >=0.03？
7. precision LCB 能否从 0.673827 拉到 >=0.75？
8. bad-event UCB 能否从 0.154558 压到 <=0.05？
9. true delta compute v10 能否继续压到 step <=1.50？
10. system-legal controller 能否 LDO/LSO？
11. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.65 的结果将给出清晰分叉：

```text
if oracle infeasible:
  reset safe-useful target / event definition.

if oracle feasible but legal frontier fail:
  repair joint sufficient statistics.

if reference frontier pass but compute fail:
  kernelization remains blocker.

if reference and compute both pass:
  open system-legal controller, LDO/LSO, paired replay.

if LDO/LSO fail:
  no dataset-specific tuning; repair dataset-agnostic support/family reliability.
```
