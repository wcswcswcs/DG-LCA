# DG-KAN v9.2.66 Oracle-Legal Gap Closure 与 Borderline Risk Calibration 完整实验计划

> 本计划基于 v9.2.65 `Joint Safe-Useful Feasibility 与 Null-Bad Conflict Resolution` 的真实复盘制定。  
> v9.2.65 的 terminal route 是：
>
> ```text
> route = R14-JointSafeUsefulStillTiny
> base_candidate = LQ-t2-h256
> success_v9265_strict_purekan_functional = False
> success_v9265_full_functional = False
> success_v9265_external_ready = False
> ```
>
> v9.2.65 的关键事实是：
>
> ```text
> P1 oracle-bounded safe-useful feasibility:
>   oracle_safe_useful_feasible = 1
>   oracle_precision = 1.0
>   oracle_coverage = 0.067336
>   oracle_bad_event = 0.0
>   oracle_null_rate = 0.0
>   oracle_support_balanced_pass = 1
>   accepted_signal_strata_count = 37
>   accepted_family_count = 213
>
> P2 null-bad conflict:
>   null_bad_conflict_detected = 1
>   P_bad_given_low_null = 0.573923
>   P_null_given_low_bad = 0.748132
>   P_safe_useful_given_low_null_low_bad = 0.553995
>   major_failure_attribution_fraction = 1.0
>
> P3 four-class joint statistic:
>   best = JC4-HorizonPersistentSafeUseful
>   safe-useful AUC = 0.836857
>   precision = 0.839623
>   coverage = 0.011684
>   bad-event = 0.066038
>   null-rate = 0.028302
>   precision LCB = 0.758099
>   bad-event UCB = 0.130077
>   utility/confidence pass = 0
>
>   wider candidates:
>     JC1 coverage = 0.117063, bad-event = 0.026365, null-rate = 0.580979, precision = 0.347458
>     JC2 coverage = 0.071098, bad-event = 0.043411, null-rate = 0.347287, precision = 0.541085
>     JC3 coverage = 0.116292, bad-event = 0.022749, null-rate = 0.614218, precision = 0.343128
>     JC5 coverage = 0.104497, bad-event = 0.029536, null-rate = 0.539030, precision = 0.385021
>
> P4 family-stable joint support:
>   pass = 1
>   natural rows = 24192
>   balanced diagnostic rows = 6000
>   signal strata = 318
>   family = 2730
>   duplicate rows = 0
>   accepted support pass = 1
>
> P5 exact-reference frontier:
>   best = C4-HorizonPersistentSafeUsefulController
>   precision = 0.839623
>   coverage = 0.011684
>   bad-event = 0.066038
>   null-rate = 0.028302
>   precision LCB = 0.758099
>   bad-event UCB = 0.130077
>   exact_reference_deployable = 0
>
>   broad borderline C0:
>   precision = 0.773292
>   coverage = 0.035494
>   bad-event = 0.055901
>   null-rate = 0.133540
>   precision LCB = 0.724493
>   bad-event UCB = 0.086624
>
> P6 true-delta compute:
>   best = TBD0-V9256CBD0Reference
>   true-delta AUC = 0.879072
>   safe-useful AUC = 0.798460
>   agreement = 1.0
>   step_ratio_q90 = 2.863280 > 1.50
>   true_delta_compute_pass = 0
>
> current blocker:
>   exact_reference_still_tiny_after_joint_model
>
> next_required_implementation:
>   repair_joint_safe_useful_decision_geometry
> ```
>
> v9.2.66 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.65 已经证明 target 上界存在；现在不是 target infeasible，而是 legal frontier 没有追上 oracle frontier。}
> }
> $$
>
> 因此 v9.2.66 不应该继续泛泛地加一个 score，也不应该直接转入 kernel-only。  
> 本轮的本质任务是：
>
> $$
> \boxed{
> \text{定位 oracle frontier 与 legal frontier 的 gap，并把 C0 broad-borderline 与 C4 precise-tiny 两条边界桥接起来。}
> }
> $$
>
> v9.2.65 的结果显示存在两个相反的 legal 边界：
>
> ```text
> C0:
>   coverage 足够，null-rate 接近可接受，但 bad-event / confidence 不过。
>
> C4:
>   precision 和 null-rate 好，precision LCB 也过，但 coverage 太小，bad-event/UCB 不过。
> ```
>
> 所以 v9.2.66 不能再把所有问题都归为 “joint score 不够强”。  
> 现在要把问题拆成：
>
> ```text
> 1. C0 为什么只差一点 bad-event / confidence？
> 2. C4 为什么丢掉了大量 oracle safe-useful rows？
> 3. Oracle accepted rows 中哪些 legal features 已经可观测，哪些不可观测？
> 4. 是否存在 C0 的 surgical bad-risk trim，而不是整体 threshold 收紧？
> 5. 是否存在 C4 的 oracle-neighbor coverage recovery，而不是整体 threshold 放宽？
> 6. 如果 legal reference finally deployable，true-delta compute 是否成为唯一 blocker？
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
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report oracle/legal gap by dataset
  report C0 bad-event excess by dataset
  report C4 coverage loss by dataset
  report family / stratum / horizon failure modes by dataset
  report leave-dataset-out matrices
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: threshold = A
  if dataset == KMNIST: use controller B
  if dataset == MNIST: skip risky rows
  tune threshold per dataset
  choose support family per dataset
  choose controller per dataset
  choose selected/full-logit mode per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-null-support-family-level / branch-delta features。Dataset name 只能进入 report，不能进入 commit rule。

---

# Part I. 对 v9.2.65 的独立判断

## 1. v9.2.65 没有达到目标

v9.2.65 没有 strict PureKAN functional success。失败不在 paired replay，而在更早的 exact-reference frontier 和 true-delta compute：

```text
exact_reference_deployable = 0
true_delta_compute_pass = 0
system_legal_controller = not_run
LDO / LSO / paired replay = not_run
```

P5 的 best exact-reference controller：

```text
C4-HorizonPersistentSafeUsefulController
precision = 0.839623
coverage = 0.011684
bad-event = 0.066038
null-rate = 0.028302
precision LCB = 0.758099
bad-event UCB = 0.130077
```

它不能 official，因为：

```text
coverage < 0.03
bad-event > 0.05
bad-event UCB > 0.05
```

这个失败不是小数点问题。Coverage 只有 `0.011684`，距离下限 `0.03` 还差：

$$
0.03-0.011684=0.018316.
$$

需要大约：

$$
\frac{0.03}{0.011684}\approx2.57
$$

倍 coverage expansion，同时还要把 bad-event 从 `0.066038` 压到 `<=0.05`，把 bad-event UCB 从 `0.130077` 压到 `<=0.05`。

## 2. v9.2.65 的真实进展

v9.2.65 是最近几轮中非常重要的一轮，因为它第一次把 “target 是否存在” 和 “legal controller 是否能找回 target” 分开了。

第一，oracle-bounded frontier 通过：

```text
oracle precision = 1.0
oracle coverage = 0.067336
oracle bad-event = 0.0
oracle null-rate = 0.0
support-balanced pass = 1
accepted strata = 37
accepted families = 213
```

这说明当前 rows / labels / support 中确实存在一个满足 deployable coverage 的 safe-useful 上界。也就是说，v9.2.65 排除了一个最坏可能：不是当前 target 完全 infeasible。

第二，null-bad conflict 被测出来：

```text
P_bad_given_low_null = 0.573923
P_null_given_low_bad = 0.748132
P_safe_useful_given_low_null_low_bad = 0.553995
```

这说明 null 和 bad 不是两个可独立串联的 clean axes。顺序 gate 之所以反复坍缩，是因为 low-null 区经常 high-bad，low-bad 区经常 high-null。

第三，four-class joint model 有 signal。JC1/JC2/JC3/JC5 的 AUC 都很高，JC4 能把 precision 和 null-rate 做好。问题不是完全没有可观测信号，而是不同 candidate 分别满足不同子目标，却没有一个同时满足 precision / coverage / bad-event / null / confidence。

第四，support 已经不是第一 blocker。P4 rows / family / strata / duplicate / accepted support 全部过关。继续盲目扩 support rows 不会直接解决 P5。

## 3. v9.2.65 的真实失败

v9.2.65 的失败是 **oracle/legal gap**，而不是 “没有 safe-useful support”。

具体来说，P1 oracle：

$$
Coverage_{\text{oracle}}=0.067336,
$$

P5 best legal exact-reference：

$$
Coverage_{\text{legal-best}}=0.011684.
$$

两者 gap 是：

$$
0.067336-0.011684=0.055652.
$$

换成比例：

$$
\frac{0.011684}{0.067336}\approx0.1735.
$$

当前 legal best 只找回了 oracle coverage 的约 `17.35%`。这才是核心问题。

更关键的是，P5 中还有一个 broad borderline candidate：

```text
C0:
  precision = 0.773292
  coverage = 0.035494
  bad-event = 0.055901
  null-rate = 0.133540
  precision LCB = 0.724493
  bad-event UCB = 0.086624
```

C0 已经满足 coverage，null-rate 也在 `0.15` 内，precision point estimate 也高于 `0.75`，但 bad-event 稍微超过 `0.05`，LCB/UCB 不稳。这说明可部署 region 可能并非很远；问题是如何对 C0 做 **surgical risk trim**，而不是整体收紧到 C4 那种 tiny slice。

同时，C4：

```text
C4:
  precision = 0.839623
  coverage = 0.011684
  bad-event = 0.066038
  null-rate = 0.028302
```

C4 的 null 和 precision 更好，但 coverage 太低且 bad-event 仍超。它需要 **oracle-neighbor coverage recovery**，而不是简单降低 threshold。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{oracle-legal gap not closed: broad legal region is borderline unsafe, precise legal region is too small.}
}
$$

具体包括：

```text
1. Oracle deployable frontier exists。
2. Legal joint statistics can rank SU, but not recover enough oracle support。
3. Null-bad conflict makes sequential gating collapse。
4. C0 is broad but borderline unsafe/confidence-fail。
5. C4 is precise but tiny and still bad-UCB fail。
6. Support family/strata are broad enough。
7. True-delta compute remains too expensive but is not the only blocker。
```

所以 v9.2.66 的主线不应是 “再造一个 global joint score”，而应是：

```text
A. oracle/legal gap localization；
B. C0 surgical bad-event trim；
C. C4 oracle-neighbor coverage recovery；
D. confidence calibration and family LCB/UCB repair；
E. only if exact-reference deployable, promote to true-delta system lane。
```

## 5. 是否还在正确道路上

是，而且 v9.2.65 是方向正确的强证据。因为 P1 oracle feasibility 告诉我们：目标不是虚构的，support 里有可部署上界。现在要解决的是 legal observability / decision geometry，而不是回到 basis sweep、dataset tuning 或 PureKANConv。

正确路线：

```text
oracle safe-useful feasible
→ oracle/legal gap localization
→ broad-borderline C0 repair
→ precise-tiny C4 expansion
→ exact-reference deployable frontier
→ true-delta compute system pass
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线：

```text
继续只加一个 AUC 更高的 joint score；
继续只硬化 bad-event gate；
继续只放宽 support threshold；
继续只做 kernelization；
把 oracle pass 写成 functional success；
把 C0 point estimate 写成 near-pass；
按 dataset 调 controller。
```

---

# Part II. v9.2.66 总体目标

v9.2.66 的总体目标是：

$$
\boxed{
\text{闭合 oracle/legal gap，把 existing oracle-safe frontier 转化为 legal exact-reference deployable frontier。}
}
$$

本轮必须并行回答八个问题：

```text
Q1:
  Oracle accepted rows 中，有多少被 C0 接受？有多少被 C4 接受？有多少被两者都拒绝？

Q2:
  C0 的 bad-event excess 来自哪些 family / stratum / horizon / score-conflict？

Q3:
  C4 丢掉的 oracle-safe rows 是因为 horizon-persistence 太硬、family UCB 太保守，还是 value score bucket 错？

Q4:
  是否存在 C0 surgical trim：
    在 coverage >= 0.03 下把 bad-event <= 0.05 且 UCB <= 0.05？

Q5:
  是否存在 C4 oracle-neighbor expansion：
    在 precision/null 不崩的前提下把 coverage 拉到 >= 0.03？

Q6:
  C0 与 C4 的 union / intersection / staged controller 是否能优于单一 controller？

Q7:
  exact reference deployable 后，true-delta compute 是否成为唯一 blocker？

Q8:
  如果 exact reference 仍不 deployable，是否是 legal feature 不足，还是 confidence calibration 过保守？
```

v9.2.66 的核心 stop-go 是：

$$
\boxed{
\text{CBD0 exact-reference + legal features 是否能达到 deployable frontier？}
}
$$

如果不能，下一步不是 kernel-only，而是 redesign legal sufficient statistics / output-delta features。  
如果能，下一步才进入 true-delta compute closure and official system controller。

---

# Part III. 核心假设

## H1：oracle/legal gap 主要来自 C4 coverage loss，而不是 target infeasibility

v9.2.65 已证明 oracle feasible：

$$
Coverage_{\text{oracle}}=0.067336.
$$

H1 认为 C4 丢掉的 oracle-safe rows 可以通过 neighborhood / family / score-threshold repair 找回。

H1 成立标准：

C4 expansion 后：

$$
Coverage_{\text{heldout}}\geq0.03,
$$

同时：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
NullRate_{\text{heldout}}\leq0.15.
$$

H1 失败标准：

C4 expansion 一旦 coverage 超过 `0.02`，precision 或 bad-event 立即崩溃。

## H2：C0 是可部署 frontier 的 broad seed，只需要 surgical bad-event trim

C0 已有：

```text
coverage = 0.035494
precision = 0.773292
null-rate = 0.133540
bad-event = 0.055901
```

它已经非常接近 point-estimate gate。H2 认为 C0 的失败来自少数 risk pockets，而不是整个 broad region 都不可用。

H2 成立标准：

在 C0 accepted region 内，找到 legal trim statistic，使：

$$
Coverage_{\text{trim}}\geq0.03,
$$

$$
BadEventRate_{\text{trim}}\leq0.05,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

H2 失败标准：

任何 bad-event trim 都把 coverage 降到 `<0.025`，或 bad-event UCB 仍 `>0.05`。

## H3：null-bad conflict 需要局部 joint resolution，而不是 global sequential gates

P2 已检测：

$$
P(Bad\mid LowNull)=0.573923,
$$

$$
P(Null\mid LowBad)=0.748132.
$$

H3 认为 null-bad conflict 在不同 family / horizon / stratum 中形式不同，需要 local conflict resolver。

H3 成立标准：

local conflict resolver 相比 global joint score 至少改善：

$$
\Delta BadEventRate\leq-0.02
$$

or:

$$
\Delta NullRate\leq-0.05
$$

同时 coverage 不低于 `0.03`。

H3 失败标准：

local conflict features 与 global score 等价，无法改善 C0/C4 tradeoff。

## H4：legal-oracle gap 有可解释 feature-missing modes

H4 认为被 oracle accept 但 legal reject 的 rows 能归入少数 modes：

```text
M1-horizon-persistence-overreject
M2-family-UCB-overconservative
M3-value-density-underestimated
M4-null-conflict-overpenalized
M5-bad-risk-overpenalized
M6-support-neighbor-missed
M7-delta-feature-missing
M8-label-conflict-ambiguous
```

H4 成立标准：

oracle legal gap attribution fraction：

```text
>= 0.90
```

且至少一个 mode 的 targeted repair 能提升 legal-oracle overlap：

$$
Jaccard_{\text{legal,oracle}}\geq0.30.
$$

H4 失败标准：

oracle accepted rows 在 legal feature space 中没有可解释聚类或 mode。

## H5：true-delta compute 仍是并行 blocker，但必须等 reference frontier 可部署后才成为主 blocker

当前：

$$
StepRatio_{q90}=2.863280>1.50.
$$

H5 成立标准：

至少一个 true-delta v11 candidate 达到：

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

true-delta compute 仍过慢或 signal lost。  
但若 H1-H4 不成立，即使 H5 成立，也不得进入 official functional success。

---

# Part IV. 新的决策设计：从 global joint score 到 bridge frontier

## 1. Oracle/legal gap decomposition

定义 oracle accepted set：

$$
A_O=\{e:Y_{SU}(e)=1\}.
$$

定义 C0 accepted set：

$$
A_{C0}.
$$

定义 C4 accepted set：

$$
A_{C4}.
$$

需要分解：

$$
A_O\cap A_{C0},
$$

$$
A_O\cap A_{C4},
$$

$$
A_O\setminus(A_{C0}\cup A_{C4}),
$$

$$
A_{C0}\setminus A_O,
$$

$$
A_{C4}\setminus A_O.
$$

每个区域都必须记录：

```text
precision
coverage
bad-event
null-rate
family count
strata count
max family share
legal feature distribution
```

## 2. C0 surgical trim

C0 不是废弃对象，而是 broad seed。定义：

$$
A_{C0}^{trim}
=
A_{C0}
\setminus
R_{\text{bad-pocket}}.
$$

其中 bad pocket 由 local risk / family conflict 决定：

$$
R_{\text{bad-pocket}}
=
\{e:
UCB_{\text{bad-local}}(e)>\rho_b
\land
SafeUsefulDensity(e)<d_s
\}.
$$

目标不是最大化 AUC，而是最小删除：

$$
\min |R_{\text{bad-pocket}}|
$$

subject to：

$$
BadEventRate(A_{C0}^{trim})\leq0.05,
$$

$$
BadEventRate_{\text{UCB}}(A_{C0}^{trim})\leq0.05,
$$

$$
Coverage(A_{C0}^{trim})\geq0.03.
$$

## 3. C4 oracle-neighbor expansion

C4 是 precise seed。定义：

$$
A_{C4}^{expand}
=
A_{C4}
\cup
N_{\text{oracle-like}}(A_{C4}).
$$

其中 neighborhood 只能用 legal features：

$$
N_{\text{oracle-like}}(A_{C4})
=
\{e:
d_{\psi}(e,A_{C4})\leq\epsilon
\land
Rel_{\text{family}}(e)\geq r
\land
ConflictPenalty(e)\leq\pi
\}.
$$

距离：

$$
d_{\psi}(e,A)
=
\min_{a\in A}
\left\|
\Sigma^{-1/2}(\psi(e)-\psi(a))
\right\|_2.
$$

Feature vector $\psi(e)$：

```text
joint SU score
RU UCB
HN UCB
BN UCB
null-bad conflict score
horizon persistence
family safe-useful LCB
support density
delta efficiency
control gap LCB
```

## 4. Bridge controller

最终 bridge frontier 由三部分组成：

$$
A_{\text{bridge}}
=
A_{C0}^{trim}
\cup
A_{C4}^{expand}
\cup
A_{\text{oracle-proxy-pocket}}.
$$

其中 oracle-proxy-pocket 不是 posthoc oracle，而是 legal approximation：

$$
A_{\text{oracle-proxy-pocket}}
=
\{e:
S_{\text{joint}}(e)\geq\tau
\land
LocalConflictPenalty(e)\leq\pi
\land
FamilyStable(e)
\}.
$$

最终 accept：

$$
Accept(e)=\mathbb{1}[e\in A_{\text{bridge}}].
$$

Calibration objective：

$$
\max Coverage(A_{\text{bridge}})
$$

subject to：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
NullRate\leq0.15,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
accepted\_signal\_strata\_count\geq5,
$$

$$
accepted\_family\_count\geq32.
$$

---

# Part V. Candidate designs

## 1. Gap localization statistics

### G0：C0/C4 reference sets

Reference only. Reconstruct v9.2.65 C0 and C4 accept sets.

### G1：OracleMissModeClassifier

Classify oracle accepted but legal rejected rows:

```text
M1-horizon-persistence-overreject
M2-family-UCB-overconservative
M3-value-density-underestimated
M4-null-conflict-overpenalized
M5-bad-risk-overpenalized
M6-support-neighbor-missed
M7-delta-feature-missing
M8-label-conflict-ambiguous
```

### G2：LegalFalsePositiveModeClassifier

Classify legal accepted but oracle rejected rows:

```text
F1-risky-useful
F2-harmless-null
F3-bad-null
F4-family-conflict
F5-horizon-fragile
F6-support-edge
F7-score-artifact
```

### G3：OracleNeighborDensity

$$
D_O(e)
=
\frac{k_O(e)}
{k(e)+\epsilon},
$$

where $k_O(e)$ is the number of oracle accepted neighbors in legal feature space.  
Diagnostic only if directly using oracle label; official version uses calibrated family safe-useful LCB.

## 2. C0 trim statistics

### T0：C0 reference

Reference only.

### T1：LocalBadPocketUCB

$$
UCB_{\text{bad-local}}(e)
=
\hat{p}_{bad}(\mathcal{N}(e))
+
\kappa
\sqrt{
\frac{\hat{p}_{bad}(1-\hat{p}_{bad})}{n_{\mathcal{N}(e)}+\epsilon}
}.
$$

### T2：ConflictRiskTrim

$$
Penalty_{\text{conflict}}(e)
=
\alpha_1 I[LowNull(e)]UCB_{\text{bad}}(e)
+
\alpha_2 I[LowBad(e)]UCB_{\text{null}}(e)
+
\alpha_3 I_{\text{label-conflict}}(e).
$$

### T3：FamilyBadSpikeTrim

Reject families whose bad-event UCB spikes inside C0:

$$
Trim(e)=
\mathbb{1}[
UCB_{\text{bad}}(family(e)\cap A_{C0})>\rho_f
].
$$

### T4：HorizonFragilityTrim

Reject if short-horizon value is high but long-horizon value is unstable:

$$
S_{\text{fragile}}
=
\max(0,Gap_{20})
+
\max(0,Gap_{80})
-
\min(Gap_{240},Gap_{640}).
$$

### T5：MinimalDeletionConstrainedTrim

Search trim combinations and choose the one with minimum coverage loss subject to official constraints.

## 3. C4 expansion statistics

### E0：C4 reference

Reference only.

### E1：OracleNeighborExpansionDiagnostic

Posthoc diagnostic: how much coverage could be recovered by adding oracle-neighbor rows around C4.

### E2：LegalNeighborExpansion

Add rows close to C4 in legal feature space:

$$
d_\psi(e,A_{C4})\leq\epsilon.
$$

### E3：FamilyBackoffExpansion

Add rows from parent families if parent safe-useful LCB passes:

$$
LCB_{\text{SU}}(parent(f))\geq r_p.
$$

### E4：HorizonRelaxedExpansion

Relax horizon persistence only if family and local bad UCB are safe:

$$
Persist(e)\geq\tau_{relaxed}
\land
UCB_{\text{bad-local}}(e)\leq\rho_b.
$$

### E5：DensityBalancedExpansion

Expansion candidate must not collapse family/strata balance:

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

## 4. Bridge controllers

### C0：v9.2.65 broad reference

Diagnostic reference.

### C1：v9.2.65 precise reference

Diagnostic reference.

### C2：C0-MinimalBadTrim

$$
Accept=A_{C0}^{trim}.
$$

### C3：C4-LegalNeighborExpansion

$$
Accept=A_{C4}^{expand}.
$$

### C4：C0TrimPlusC4Expansion

$$
Accept=A_{C0}^{trim}\cup A_{C4}^{expand}.
$$

### C5：BridgeFrontierController

$$
Accept=A_{C0}^{trim}\cup A_{C4}^{expand}\cup A_{\text{oracle-proxy-pocket}}.
$$

### C6：ConstrainedBridgeOptimizer

Search over trim / expansion / bridge thresholds:

$$
\max Coverage
$$

subject to official constraints.

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.65 boundary reproduction

### 目标

确认 v9.2.65 boundary 稳定。尤其是 oracle feasible、C0/C4 frontier、P5 fail、P6 compute fail。

### 必须记录

```text
route
source_route_v9264
oracle_safe_useful_feasible
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
null_bad_conflict_detected
P_bad_given_low_null
P_null_given_low_bad
joint_stat_pass
best_joint_stat_id
exact_reference_deployable
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
support_joint_pass
true_delta_compute_pass
true_delta_step_ratio_q90
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R14-JointSafeUsefulStillTiny
oracle_safe_useful_feasible = 1
support_joint_pass = 1
exact_reference_deployable = 0
true_delta_compute_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_oracle_vs_legal_gap_ladder.svg
p0_C0_C4_tradeoff.svg
p0_decision_compute_gate_ladder.svg
```

---

## P1：oracle/legal gap localization

### 目标

定位 oracle accepted rows 为什么没有被 legal frontier 找回。

### 必须记录

```text
row_id
split_id
dataset
seed
horizon
signal_stratum
event_family
oracle_accept
C0_accept
C4_accept
C0_trim_candidate
C4_expand_candidate
safe_useful
risky_useful
harmless_null
bad_null
bad_event
null_event
joint_scores
conflict_scores
support_scores
gap_mode
false_positive_mode
false_negative_mode
```

必须输出区域统计：

```text
A_oracle_count
A_C0_count
A_C4_count
A_oracle_intersect_C0_count
A_oracle_intersect_C4_count
A_oracle_missed_by_both_count
A_C0_false_positive_count
A_C4_false_positive_count
legal_oracle_jaccard_C0
legal_oracle_jaccard_C4
```

### 判断标准

P1 pass：

```text
oracle_miss_attribution_fraction >= 0.90
legal_false_positive_attribution_fraction >= 0.90
at least one repairable oracle-miss mode has count >= 0.01 coverage
```

### 可视化

```text
p1_oracle_legal_venn.svg
p1_oracle_miss_mode_sankey.svg
p1_legal_false_positive_mode_sankey.svg
p1_legal_feature_space_umap.svg
p1_oracle_gap_by_family_stratum.svg
```

---

## P2：C0 surgical bad-event trim

### 目标

从 C0 broad borderline region 出发，最小删除 bad pockets，使 C0 达到 deployable gate。

### 必须记录

```text
trim_id
base_controller = C0
features_used
thresholds
rows_removed
coverage_before
coverage_after
precision_before
precision_after
bad_event_before
bad_event_after
null_rate_before
null_rate_after
precision_lcb
bad_event_ucb
removed_mode_distribution
removed_family_distribution
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

C0 trim pass：

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

Confidence pass：

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

Minimal deletion diagnostic：

$$
Coverage_{\text{after}}\geq0.85\cdot Coverage_{C0}.
$$

### 可视化

```text
p2_C0_trim_precision_coverage_bad.svg
p2_minimal_deletion_curve.svg
p2_removed_bad_pocket_heatmap.svg
p2_trim_threshold_surface.svg
```

---

## P3：C4 oracle-neighbor coverage recovery

### 目标

从 C4 precise tiny region 出发，找回 oracle-like legal neighbors，使 coverage 到达 deployable range。

### 必须记录

```text
expansion_id
base_controller = C4
features_used
distance_metric
thresholds
rows_added
coverage_before
coverage_after
precision_before
precision_after
bad_event_before
bad_event_after
null_rate_before
null_rate_after
precision_lcb
bad_event_ucb
added_mode_distribution
added_family_distribution
oracle_overlap_gain
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

C4 expansion pass：

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

Oracle-overlap diagnostic：

$$
Jaccard_{\text{legal,oracle}}\geq0.30.
$$

Support balance：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p3_C4_expansion_frontier.svg
p3_oracle_neighbor_recovery.svg
p3_added_rows_mode_heatmap.svg
p3_expansion_distance_curve.svg
```

---

## P4：bridge controller

### 目标

组合 C0 trim 与 C4 expansion，判断 bridge frontier 是否能过 exact-reference deployable gate。

### 必须记录

```text
bridge_controller_id
C0_trim_id
C4_expansion_id
oracle_proxy_pocket_id
thresholds
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
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
legal_oracle_jaccard
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Bridge pass：

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

### 可视化

```text
p4_bridge_frontier_precision_coverage_bad_null.svg
p4_C0_C4_bridge_ladder.svg
p4_bridge_oracle_overlap.svg
p4_bridge_support_balance.svg
```

---

## P5：exact-reference deployable frontier v10

### 目标

在 CBD0 exact reference 上确认最终 legal bridge frontier 是否可部署。P5 是本轮最重要 gate。

### 必须记录

```text
controller_id
bridge_controller_id
joint_stat_id
trim_stat_id
expansion_stat_id
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
oracle_gap_remaining
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
R10-C0BadTrimFail
R11-C4ExpansionFail
R12-OracleLegalGapUnresolved
R13-BridgeStillTiny
R14-ReferenceStillNotDeployableAfterBridge
```

### 可视化

```text
p5_reference_frontier_v10_precision_coverage_bad_null.svg
p5_oracle_legal_gap_after_bridge.svg
p5_deployable_region_ladder.svg
p5_lcb_ucb_confidence_frontier.svg
```

---

## P6：true-delta compute v11 parallel lane

### 目标

继续推进 system path，但不让 compute lane 掩盖 decision feasibility。若 P5 过，P6 变为 primary blocker；若 P5 不过，P6 仅作 parallel diagnostic。

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
AUC_bridge_accept
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
joint_score_component_time
trim_component_time
expansion_component_time
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
p6_true_delta_v11_cost_signal_pareto.svg
p6_compute_vs_decision_ladder.svg
p6_residual_subphase_after_bridge.svg
```

---

## P7：system-legal exact-signal controller

### 目标

只有 P5 reference deployable pass 与 P6 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
bridge_controller_id
joint_stat_id
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
bridge_controller_id
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
bridge_controller_id = best P8 survivor
joint_stat_id = best P8 survivor
support_stat_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledBridgeController, ShuffledJointScore,
           ShuffledTrimScore, ShuffledExpansionScore, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledSignalChannel, FunctionalChannelShuffled, TailMaskShuffled,
           RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled,
           InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
bridge_controller_id
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
ShuffledBridgeController = fail
ShuffledJointScore = fail
ShuffledTrimScore = fail
ShuffledExpansionScore = fail
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

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9266.csv
p0_v9265_boundary_reproduction.csv
p1_oracle_legal_gap_localization.csv
p2_C0_surgical_bad_event_trim.csv
p3_C4_oracle_neighbor_coverage_recovery.csv
p4_bridge_controller.csv
p5_exact_reference_deployable_frontier_v10.csv
p6_true_delta_compute_v11_parallel_lane.csv
p7_system_legal_exact_signal_controller.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
oracle_legal_gap_trace_v9266.csv
C0_trim_trace_v9266.csv
C4_expansion_trace_v9266.csv
bridge_frontier_trace_v9266.csv
reference_frontier_v10_trace.csv
true_delta_compute_v11_trace.csv
system_controller_trace_v9266.csv
leaveout_trace_v9266.csv
paired_replay_branch_trace_v9266.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9265_boundary_unstable
F3_dataset_tuning_detected
F4_oracle_legal_gap_unattributed
F5_C0_bad_trim_fail
F6_C0_trim_coverage_loss
F7_C4_expansion_fail
F8_C4_expansion_bad_event_fail
F9_bridge_frontier_tiny
F10_bridge_frontier_precision_fail
F11_bridge_frontier_bad_event_fail
F12_bridge_frontier_null_fail
F13_precision_lcb_fail
F14_bad_event_ucb_fail
F15_exact_reference_still_not_deployable_after_bridge
F16_true_delta_compute_still_expensive
F17_true_delta_signal_lost
F18_system_controller_precision_fail
F19_system_controller_coverage_fail
F20_system_controller_bad_event_fail
F21_system_controller_null_rate_fail
F22_leave_dataset_out_fail
F23_leave_stratum_out_fail
F24_paired_replay_control_equivalent
F25_shuffle_control_pass
F26_functional_lr_equivalent
F27_short_run_task_drop
F28_full_run_no_macro_hard_stratum_gain
F29_strong_baseline_explains_gain
F30_robustness_fail
F31_external_not_ready
F32_fake_or_proxy_violation
F33_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.65 boundary reproduced.

R2-OracleLegalGapLocalized:
  oracle/legal gap and false positives are attributed.

R3-C0SurgicalTrimPass:
  C0 broad seed becomes deployable after minimal bad-event trim.

R4-C4OracleNeighborExpansionPass:
  C4 precise seed recovers coverage without safety collapse.

R5-BridgeControllerPass:
  C0-trim + C4-expansion bridge passes heldout frontier.

R6-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + bridge frontier passes precision / coverage / bad-event / null gate.

R7-TrueDeltaComputeV11SystemPass:
  true branch-delta v11 passes system envelope while preserving signal.

R8-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-C0BadTrimFail:
  broad C0 seed cannot be made safe without losing coverage.

R13-C4ExpansionFail:
  precise C4 seed cannot recover coverage without safety collapse.

R14-OracleLegalGapUnresolved:
  oracle frontier exists but legal features cannot explain or recover it.

R15-BridgeStillTiny:
  bridge frontier still coverage < 0.03.

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
v9265_boundary_pass
dataset_tuning_detected
oracle_safe_useful_feasible
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
oracle_legal_gap_localized
oracle_miss_attribution_fraction
legal_false_positive_attribution_fraction
C0_trim_pass
C0_trim_precision
C0_trim_coverage
C0_trim_bad_event
C0_trim_null_rate
C0_trim_precision_lcb
C0_trim_bad_event_ucb
C4_expansion_pass
C4_expansion_precision
C4_expansion_coverage
C4_expansion_bad_event
C4_expansion_null_rate
C4_expansion_precision_lcb
C4_expansion_bad_event_ucb
best_bridge_controller_id
bridge_controller_pass
bridge_precision
bridge_coverage
bridge_bad_event
bridge_null_rate
bridge_precision_lcb
bridge_bad_event_ucb
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
legal_oracle_jaccard
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
true_delta_joint_auc
true_delta_bridge_auc
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
success_v9266_strict_purekan_functional
success_v9266_full_functional
success_v9266_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 oracle/legal gap localization
  P2 C0 surgical trim
  P3 C4 oracle-neighbor expansion
  P4 bridge controller
  P5 exact reference deployable frontier v10
  P6 true-delta compute v11

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
P1-P5 decision geometry can run before true-delta compute pass.
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

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.65 boundary reproduced
oracle/legal gap localized
C0 trim measured
C4 expansion measured
bridge controller measured
exact reference deployable frontier v10 measured
true-delta compute v11 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
exact reference deployable frontier pass
+
multi-stratum / multi-family support pass
+
LCB/UCB confidence pass
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
1. v9.2.65 boundary cannot be reproduced；
2. oracle/legal gap cannot be attributed；
3. C0 trim cannot reduce bad-event without coverage loss；
4. C4 expansion cannot recover coverage；
5. bridge frontier remains tiny；
6. precision LCB remains below 0.75；
7. bad-event UCB remains above 0.05；
8. null rate remains above 0.15；
9. exact reference remains not deployable after bridge repair；
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

# Part XI. 最终解释规则

## Case A：P5 reference deployable + P6 compute + P8/P9 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：P5 reference deployable, P6 compute fail

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused bridge-score compute，不调 dataset。

## Case C：P6 compute pass, P5 reference fail

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 oracle/legal gap and bridge frontier，不走 kernel-only。

## Case D：C0 trim fail

必须声明：

```text
broad legal seed is fundamentally unsafe under current features.
```

下一步修 local bad-risk sufficient statistics，而不是继续扩大 C0。

## Case E：C4 expansion fail

必须声明：

```text
precise legal seed cannot recover oracle support under current feature geometry.
```

下一步修 oracle-neighbor representation / output-delta features。

## Case F：oracle/legal gap remains unexplained

必须声明：

```text
oracle frontier exists, but current legal features cannot explain or recover it.
```

下一步重设 legal sufficient statistics，不继续 threshold tuning。

## Case G：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

---

# Part XII. 最终建议

v9.2.66 的一句话策略是：

$$
\boxed{
\text{不要再追求单个更高 AUC 的 score；先闭合 oracle/legal gap，用 C0 trim 与 C4 expansion 构造 bridge frontier。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
oracle 是否存在；
support family 是否够；
joint score 是否有 AUC；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. Oracle accepted rows 中，legal features 缺了什么？
2. C0 的 bad-event excess 是少数 pocket 还是全局问题？
3. C4 丢掉的 oracle rows 是过保守还是缺 feature？
4. 是否存在 coverage >=0.03 的 C0-trim frontier？
5. 是否存在 coverage >=0.03 的 C4-expansion frontier？
6. C0/C4 bridge 能否同时满足 precision / coverage / bad-event / null / LCB-UCB？
7. exact-reference frontier 过线后，true-delta compute 是否成为唯一 blocker？
8. system-legal controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.66 的结果将给出清晰分叉：

```text
if C0 trim or C4 expansion restores reference deployability:
  decision geometry is viable; proceed to true-delta compute and official controller.

if bridge frontier still tiny:
  current legal joint features cannot recover oracle support; redesign sufficient statistics.

if reference deployable but compute fail:
  kernelization remains blocker.

if reference and compute both pass:
  open system-legal controller, LDO/LSO, paired replay.

if LDO/LSO fail:
  no dataset-specific tuning; repair dataset-agnostic support/family reliability.
```
