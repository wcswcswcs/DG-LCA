# DG-KAN v9.2.67 Borderline-Localized Bridge Repair 与 C0/C4 Pareto Calibration 完整实验计划

> 本计划基于 v9.2.66 `Oracle-Legal Gap Closure 与 Bridge Frontier` 的真实执行结果制定。  
> v9.2.66 的 terminal route 是：
>
> ```text
> route = R12-C0BadTrimFail
> base_candidate = LQ-t2-h256
> success_v9266_strict_purekan_functional = False
> success_v9266_full_functional = False
> success_v9266_external_ready = False
> ```
>
> v9.2.66 的关键事实是：
>
> ```text
> P1 oracle-legal gap localization:
>   oracle count = 1629
>   C0 count = 930
>   C4 count = 311
>   oracle missed by both = 895
>   C0 false positive = 197
>   C4 false positive = 64
>   legal-oracle Jaccard C0 = 0.401424
>   legal-oracle Jaccard C4 = 0.145895
>   repairable oracle miss mode = M1-horizon-persistence-overreject
>   repairable oracle miss coverage = 0.017072
>   oracle miss attribution fraction = 0.960894
>   legal false-positive attribution fraction = 1.0
>
> P2 C0 surgical trim:
>   best = T2-ConflictRiskTrim
>   precision = 0.831224
>   coverage = 0.026124
>   bad-event = 0.008439
>   null-rate = 0.151899
>   precision LCB = 0.778341
>   bad-event UCB = 0.030242
>   rows removed = 177
>   accepted strata/families = 14 / 56
>   C0 trim confidence pass = 1
>   C0 trim pass = 0
>
> P3 C4 oracle-neighbor expansion:
>   best = E2-LegalNeighborExpansion
>   precision = 0.534910
>   coverage = 0.097884
>   bad-event = 0.268018
>   null-rate = 0.119369
>   Jaccard = 0.486680
>   rows added = 2388
>   overlap diagnostic pass = 1
>   C4 expansion pass = 0
>
> P4/P5 bridge / reference frontier:
>   best route-level row = C0-v9265BroadReference
>   precision = 0.773292
>   coverage = 0.035494
>   bad-event = 0.055901
>   null-rate = 0.133540
>   precision LCB = 0.724493
>   bad-event UCB = 0.086624
>   exact_reference_deployable = 0
>   legal-oracle Jaccard = 0.391509
>   oracle gap remaining = 0.026565
>
> P6 true-delta compute:
>   best = TBD0-V9256CBD0Reference
>   true_delta_auc = 0.879072
>   true_delta_safe_useful_auc = 0.798460
>   true_delta_joint_auc = 0.798460
>   true_delta_bridge_auc = 0.817943
>   true_delta_conditional_bad_auc = 0.526523
>   true_delta_conditional_null_auc = 0.500210
>   agreement = 1.0
>   step_ratio_q90 = 2.863280
>   memory_ratio = 0.969501
>   true_delta_compute_pass = 0
> ```
>
> v9.2.67 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.66 已经把 problem 从 global score failure 收缩到局部边界修复问题。}
> }
> $$
>
> 现在不应该再做“再加一个更高 AUC 的 score”。  
> v9.2.66 已经显示：
>
> ```text
> C0: coverage 足够，null-rate 足够接近，但 bad-event / confidence 略不过。
> T2: safety/confidence 已过，但 coverage 少一点、null-rate 略超。
> C4: precision seed 小而干净，E2 能扩 coverage，但扩张太脏。
> Bridge: 没有把 C0 trim 与 C4 expansion 组合成 deployable frontier。
> ```
>
> 因此 v9.2.67 的本质任务是：
>
> $$
> \boxed{
> \text{用局部化的 trim / backfill / expansion / confidence calibration，把 C0/T2/C4/E2 四个边界拼成 deployable exact-reference frontier。}
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

允许按 dataset 诊断：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report C0 false-positive by dataset
  report T2 over-trim by dataset
  report C4 expansion contamination by dataset
  report oracle/legal gap by dataset
  report leave-dataset-out matrices
```

禁止按 dataset 调参：

```text
forbidden:
  if dataset == Fashion: threshold = A
  if dataset == KMNIST: trim rule = B
  if dataset == MNIST: expansion rule = C
  tune threshold per dataset
  choose controller per dataset
  choose support family per dataset
  choose selected/full-logit mode per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

Official controller 只能使用 train-stream / event-level / value-risk-null-support-family-level / branch-delta features。Dataset name 只能进入 report，不能进入 commit rule。

---

# Part I. 对 v9.2.66 的独立判断

## 1. v9.2.66 没有达到目标

v9.2.66 没有达到 strict PureKAN functional success。失败发生在三道前置门：

```text
C0_trim_pass = 0
exact_reference_deployable = 0
true_delta_compute_pass = 0
```

P7-P10 正确保持 `not_run`。因此不能声明：

```text
strict PureKAN local causal evidence
system-legal functional controller
official paired replay success
full functional success
external-ready success
```

尤其不能把下面这些写成成功：

```text
oracle feasibility
oracle-legal gap localization pass
C0 trim confidence pass
C4 expansion overlap diagnostic pass
true-delta reference AUC
C0 broad reference point estimate
```

这些都是重要诊断，不是 official controller。

## 2. v9.2.66 的真实进展

v9.2.66 的进展很实在。它不是简单失败，而是把大问题拆成了两个几乎互补的边界。

第一，oracle/legal gap 已经定位。oracle accepted rows 有 `1629`，C0 找到 `930`，C4 找到 `311`，但 oracle missed by both 仍有 `895`。P1 的 oracle miss attribution fraction 达到 `0.960894`，legal false-positive attribution fraction 达到 `1.0`，主 repairable mode 是 `M1-horizon-persistence-overreject`。这说明 legal frontier 的问题不再是黑箱。

第二，C0 trim 是真实推进。`T2-ConflictRiskTrim` 把 bad-event 从 C0 的 `0.055901` 降到 `0.008439`，bad-event UCB 从 `0.086624` 降到 `0.030242`，precision 提到 `0.831224`，precision LCB 提到 `0.778341`。这说明 C0 broad seed 并非整体不可用。

第三，C4 expansion 是真实推进。`E2-LegalNeighborExpansion` 把 coverage 从 C4 reference 的 `0.011684` 提到 `0.097884`，Jaccard 从 `0.153448` 提到 `0.486680`。这说明 C4 周围确实有大量 oracle-like recoverable coverage。

第四，true-delta signal 仍然存在。`TBD0` 的 true-delta AUC 是 `0.879072`，agreement 是 `1.0`，memory ratio 是 `0.969501`。当前不是 signal 消失，而是 system step ratio 仍是 `2.863280`，且 reference frontier 仍未 deployable。

## 3. v9.2.66 的真实失败

v9.2.66 的失败不是“oracle/legal gap 没定位”，而是：

$$
\boxed{
\text{C0 trim 与 C4 expansion 各自解决一半问题，但没有被桥接成一个 deployable controller。}
}
$$

C0 broad reference：

```text
coverage = 0.035494
precision = 0.773292
bad-event = 0.055901
null-rate = 0.133540
precision LCB = 0.724493
bad-event UCB = 0.086624
```

C0 的 coverage 与 null-rate 已在官方范围附近，bad-event 只超：

$$
0.055901-0.05=0.005901.
$$

bad-event UCB 超：

$$
0.086624-0.05=0.036624.
$$

precision LCB 差：

$$
0.75-0.724493=0.025507.
$$

T2 trim：

```text
coverage = 0.026124
precision = 0.831224
bad-event = 0.008439
null-rate = 0.151899
precision LCB = 0.778341
bad-event UCB = 0.030242
```

T2 的 safety/confidence 已经很好，但 coverage 差：

$$
0.03-0.026124=0.003876.
$$

null-rate 只超：

$$
0.151899-0.15=0.001899.
$$

也就是说，T2 并不是大失败。它更像是 **over-trim**：删掉了太多 coverage，并且留下的 null composition 还略微不平衡。

C4 expansion：

```text
coverage = 0.097884
precision = 0.534910
bad-event = 0.268018
null-rate = 0.119369
Jaccard = 0.486680
```

E2 的问题相反：它能找回 coverage 和 oracle-neighbor overlap，但引入了大量 risky rows。它不能当 controller，只能当 candidate generator。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{borderline-localized bridge repair failure。}
}
$$

更细地说：

```text
1. Oracle safe-useful frontier 仍存在；
2. C0 broad seed 接近可部署，只需要局部 bad-risk trim 和 confidence repair；
3. T2 trim safety 过了，但 coverage 低一点，null-rate 高一点；
4. C4 expansion 能恢复大量 coverage，但 contamination 太高；
5. Bridge 没有实现“C0 minimal trim + T2 backfill + C4 filtered expansion”的组合；
6. true-delta compute 仍太贵，但 reference frontier 未过前不是唯一主 blocker。
```

所以 v9.2.67 的主线不是全局重做 score，而是局部修复：

```text
C0 -> trim 得更准，少删 safe-useful；
T2 -> backfill 少量 legal-safe rows，把 coverage 拉回 0.03；
C4/E2 -> 只作为 candidate generator，必须二次过滤；
Bridge -> 用 constrained fill-up，而不是 hard union。
```

## 5. 是否还在正确道路上

是。v9.2.66 的结果比 v9.2.65 更接近可部署 frontier。v9.2.65 只是知道 oracle 存在；v9.2.66 已经知道：

```text
C0 的错误是什么；
C4 的错误是什么；
T2 能修什么；
E2 能恢复什么；
Bridge 为什么没闭合。
```

这是明确进展。  
但下一步如果还继续“再造一个 score”，就会变成小修小补。正确路线必须是：把 C0/T2/C4/E2 当成四个边界对象，做可解释、可约束、可验证的 bridge frontier。

---

# Part II. v9.2.67 总体目标

v9.2.67 的总体目标是：

$$
\boxed{
\text{通过 C0 minimal-risk trim、T2 coverage backfill、C4 filtered expansion 和 constrained bridge fill-up，闭合 exact-reference deployable frontier。}
}
$$

具体目标分成六层：

```text
1. 复现 v9.2.66 boundary。
2. 对 C0 false positives / T2 removed rows / C4 added rows 做三向 autopsy。
3. 从 T2 safety core 出发，恢复最小 coverage deficit。
4. 从 E2 expansion pool 中筛出 precision-preserving subset。
5. 构造 constrained bridge fill-up，而不是简单 union / threshold。
6. 如果 exact-reference frontier 过，再把 true-delta compute 作为 primary blocker。
```

v9.2.67 的核心 stop-go 是：

$$
\boxed{
\text{T2-like safe core 能否从 coverage }0.026124\text{ 回填到 }\geq0.03\text{，同时保持 null}\leq0.15\text{ 和 bad-event UCB}\leq0.05？
}
$$

---

# Part III. 核心假设

## H1：C0 broad seed 不是整体不可用，T2 是 over-trim

H1 认为 C0 的 bad-event excess 来自局部 pockets，而不是整个 C0 region 都危险。T2 已经证明 safety/confidence 可以过，但删得过多。

H1 成立标准：

在 C0 accepted rows 内找到比 T2 更细的 trim，使：

$$
Coverage\geq0.03,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
NullRate\leq0.15.
$$

H1 失败标准：

所有 trim 只要让 bad-event UCB 过线，coverage 就必然低于 `0.03`，或 null-rate 必然高于 `0.15`。

## H2：T2 只需要小规模 legal-safe backfill

T2 的 coverage deficit 是：

$$
0.03-0.026124=0.003876.
$$

H2 认为只需要少量 backfill rows，而不是大规模重新扩张。Backfill 必须来自 legal features，不允许使用 oracle label commit。

H2 成立标准：

从 T2 removed rows 或 C4/E2 filtered pool 中回填 rows，使：

$$
Coverage\in[0.03,0.04],
$$

同时：

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
NullRate\leq0.15.
$$

H2 失败标准：

所有 backfill rows 都高 null 或高 bad，一旦 coverage 过线，bad-event UCB 或 null-rate 立即失败。

## H3：C4/E2 不是 controller，而是 high-recall candidate generator

E2 coverage 和 Jaccard 很高，但 precision/bad-event 很差。因此 H3 不把 E2 当 controller，而是当 candidate pool。

H3 成立标准：

在 E2 added rows 中找到 legal filtered subset，满足：

$$
Precision\geq0.75,
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15,
$$

并能贡献至少：

$$
\Delta Coverage \geq0.003.
$$

H3 失败标准：

E2 filtered subset 要么 tiny，要么 precision/bad-event 无法过线。

## H4：Bridge 应该是 constrained fill-up，而不是 hard union

H4 认为 v9.2.66 的 bridge 失败是组合方式粗糙。新的 bridge 先保 T2 safety core，再按 risk order 回填 rows，直到满足 coverage lower bound，且每一步都检查 LCB/UCB。

H4 成立标准：

bridge fill-up 达到：

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
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

H4 失败标准：

bridge 仍在以下两种状态间震荡：

```text
safe but coverage < 0.03
coverage pass but bad-event / confidence fail
```

## H5：true-delta compute 是并行 blocker，但不是当前唯一主 blocker

H5 成立标准：

至少一个 true-delta v12 candidate 达到：

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

true-delta compute 仍然：

$$
StepRatio_{q90}>1.50
$$

或 signal lost。  
但如果 exact-reference frontier 仍不 deployable，即使 H5 过也不能进入 official functional success。

---

# Part IV. 新的 bridge formulation

## 1. 四个基础集合

定义：

$$
A_{C0}
=
\text{C0-v9265BroadReference accepted set}.
$$

$$
A_{T2}
=
\text{T2-ConflictRiskTrim accepted set}.
$$

$$
A_{C4}
=
\text{C4-HorizonPersistentSafeUseful accepted set}.
$$

$$
A_{E2}
=
\text{E2-LegalNeighborExpansion candidate pool}.
$$

v9.2.66 中这些集合满足：

```text
C0: broad-borderline
T2: safe-but-slightly-undercoverage
C4: precise-small
E2: high-recall-contaminated
```

## 2. T2-first constrained fill-up

从安全核心开始：

$$
A_0=A_{T2}.
$$

定义 candidate pool：

$$
P_{\text{fill}}
=
(A_{C0}\setminus A_{T2})
\cup
(A_{E2}\setminus A_{C0})
\cup
N_{\text{legal}}(A_{T2}\cup A_{C4}).
$$

每个 candidate row 有 fill score：

$$
S_{\text{fill}}(e)
=
LCB_{\text{safe-useful}}(e)
-
\lambda_B UCB_{\text{bad-local}}(e)
-
\lambda_N UCB_{\text{null-local}}(e)
+
\lambda_O S_{\text{oracle-proxy-neighbor}}(e)
+
\lambda_S LCB_{\text{support}}(e).
$$

注意：`oracle-proxy-neighbor` 不能使用 posthoc oracle label；它只能使用 legal features 来近似 oracle-neighbor geometry，例如 C4/T2 legal-neighbor density、family LCB、horizon persistence、control gap LCB。

Fill-up procedure：

```text
A = A_T2
sort P_fill by S_fill descending
for row in sorted P_fill:
    tentatively add row
    recompute precision, coverage, bad-event, null-rate, LCB/UCB
    keep row only if:
        bad_event_UCB <= 0.05 + epsilon_stage
        null_rate <= 0.15 + epsilon_stage
        precision_LCB >= 0.75 - epsilon_stage
stop when coverage >= 0.03
```

最终 official candidate 必须满足 strict thresholds, not relaxed thresholds.

## 3. C0 minimal trim repair

C0 repair 不再一次性 trim。定义 multi-level trim：

$$
A_{C0}^{trim(k)}
=
A_{C0}\setminus R_k,
$$

其中 $R_k$ 是按 risk pocket ranking 删除的前 $k$ 个 rows 或 families。

目标：

$$
\min k
$$

subject to：

$$
BadEventRate_{\text{UCB}}(A_{C0}^{trim(k)})\leq0.05,
$$

$$
Precision_{\text{LCB}}(A_{C0}^{trim(k)})\geq0.75,
$$

$$
Coverage(A_{C0}^{trim(k)})\geq0.03,
$$

$$
NullRate(A_{C0}^{trim(k)})\leq0.15.
$$

如果存在 $k$，C0 trim 直接成功。  
如果不存在，C0 trim 产出 safe core，并进入 T2 fill-up。

## 4. C4 filtered expansion repair

C4 expansion 不再全量加入 E2。定义：

$$
A_{C4}^{filtered}
=
A_{C4}
\cup
\{e\in A_{E2}: Filter(e)=1\}.
$$

Filter：

$$
Filter(e)
=
\mathbb{1}[UCB_{\text{bad-local}}(e)\leq\rho_b]
\land
\mathbb{1}[UCB_{\text{null-local}}(e)\leq\rho_n]
\land
\mathbb{1}[LCB_{\text{safe-useful}}(e)\geq\tau_s]
\land
\mathbb{1}[SupportStable(e)].
$$

目标是让 filtered E2 不是 controller，而是 low-noise expansion pool。

## 5. Final bridge

最终 bridge：

$$
A_{\text{bridge}}
=
A_{\text{C0-mintrim}}
\cup
A_{\text{T2-fillup}}
\cup
A_{\text{C4-filtered}}
\cup
A_{\text{local-oracle-proxy-pocket}}.
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

## 1. Autopsy candidates

### A0：C0/T2 deletion decomposition

Classify rows removed by T2:

```text
D1-true-bad-pocket
D2-null-heavy-pocket
D3-overtrim-safe-useful
D4-support-edge
D5-horizon-overpenalized
D6-family-UCB-overconservative
D7-label-conflict-ambiguous
```

### A1：C4/E2 contamination decomposition

Classify rows added by E2:

```text
E1-safe-useful-recovered
E2-risky-useful
E3-harmless-null
E4-bad-null
E5-support-neighbor-mismatch
E6-score-artifact
E7-horizon-fragile
```

### A2：C0/C4 complementarity matrix

Record intersections:

```text
C0 ∩ C4
C0 \ C4
C4 \ C0
T2 \ C4
E2 \ C0
Oracle missed by C0/C4
```

## 2. Trim candidates

### T0：C0 reference

Diagnostic reference.

### T1：v9.2.66 T2 reference

Diagnostic reference.

### T2a：RowwiseRiskTrimFineGrid

Same family as T2 but threshold swept finer; goal is avoid overtrim.

### T2b：FamilySpikeTrimOnly

Remove only local families with bad UCB spikes:

$$
UCB_{\text{bad}}(family\cap C0)>\rho_f.
$$

### T2c：HorizonFragilityTrimOnly

Remove rows whose horizon persistence is fragile:

$$
Gap_{20/80}>0
\land
Gap_{240/640}<\tau_h.
$$

### T2d：ConflictTrimWithNullGuard

Trim bad risk but protect low-null / high-SU rows:

$$
Trim(e)=
I[BadRisk(e)>\rho_b]
\land
I[NullRisk(e)<\rho_n]
\land
I[SUProxy(e)<\tau_s].
$$

### T2e：MinimalDeletionConstrainedTrimV2

Search trim combinations and optimize minimum coverage loss.

## 3. Backfill candidates

### B0：T2 reference

Diagnostic reference.

### B1：T2RemovedSafeBackfill

Backfill selected rows from $A_{C0}\setminus A_{T2}$.

### B2：C4NeighborBackfill

Backfill rows close to C4 seed in legal feature space.

### B3：E2FilteredBackfill

Backfill a filtered subset of E2.

### B4：FamilyBackoffBackfill

Backfill rows from parent families whose safe-useful LCB passes.

### B5：CoverageDeficitKnapsack

Solve fill-up as constrained knapsack:

$$
\max \sum_e CoverageWeight(e)x_e
$$

subject to estimated bad/null/precision confidence constraints.

## 4. Expansion candidates

### X0：C4 reference

Diagnostic reference.

### X1：E2 reference

Diagnostic reference.

### X2：E2FilteredByLocalBadUCB

Filter E2 with local bad UCB.

### X3：E2FilteredBySafeUsefulDensity

Filter E2 using legal safe-useful density.

### X4：E2FilteredByHorizonPersistence

Filter E2 using persistent no-regret signal.

### X5：E2FilteredByC0Agreement

Accept E2 rows only if compatible with C0 low-risk subregion.

## 5. Bridge candidates

### C0：v9.2.66 C0 broad reference

Diagnostic reference.

### C1：v9.2.66 T2 trim reference

Diagnostic reference.

### C2：FineTrimOnly

Best C0 fine trim candidate.

### C3：T2PlusBackfill

T2 core + selected backfill.

### C4：C4FilteredExpansion

C4 + filtered expansion.

### C5：HybridBridgeFillup

C0 fine trim + T2 backfill + C4 filtered expansion.

### C6：ConstrainedParetoBridgeOptimizer

Search all trim / backfill / expansion thresholds under official constraints.

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.66 boundary reproduction

### 目标

确认 v9.2.66 boundary 稳定，尤其是 C0/T2/C4/E2 四个边界。

### 必须记录

```text
route
v9266_boundary_pass
oracle_safe_useful_feasible
oracle_coverage
C0_precision
C0_coverage
C0_bad_event
C0_null_rate
C0_precision_lcb
C0_bad_event_ucb
T2_precision
T2_coverage
T2_bad_event
T2_null_rate
T2_precision_lcb
T2_bad_event_ucb
C4_precision
C4_coverage
C4_bad_event
C4_null_rate
E2_precision
E2_coverage
E2_bad_event
E2_null_rate
bridge_controller_pass
exact_reference_deployable
true_delta_compute_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R12-C0BadTrimFail
C0 broad borderline reproduced
T2 safe undercoverage reproduced
E2 broad contaminated expansion reproduced
exact_reference_deployable = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_C0_T2_C4_E2_tradeoff.svg
p0_oracle_legal_gap_ladder.svg
```

---

## P1：C0/T2/C4/E2 三向 autopsy

### 目标

同时解释：

```text
1. T2 为什么过 safety 但丢 coverage？
2. C0 为什么 coverage 够但 bad/confidence 不够？
3. E2 为什么 coverage/Jaccard 高但 precision/bad-event 崩？
```

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
T2_accept
C4_accept
E2_accept
safe_useful
risky_useful
harmless_null
bad_null
bad_event
null_event
removed_by_T2
added_by_E2
C0_false_positive
C4_false_positive
oracle_missed_by_C0
oracle_missed_by_C4
mode_label
legal_feature_vector
```

Mode attribution：

```text
T2 removed rows:
  D1-D7

E2 added rows:
  E1-E7

C0 false positives:
  F1-risky-useful
  F2-bad-null
  F3-horizon-fragile
  F4-family-bad-spike
  F5-score-artifact

Oracle misses:
  M1-horizon-persistence-overreject
  M2-family-UCB-overconservative
  M3-value-density-underestimated
  M4-null-conflict-overpenalized
  M5-bad-risk-overpenalized
  M6-support-neighbor-missed
```

### 判断标准

P1 pass：

```text
T2_removed_attribution_fraction >= 0.90
E2_added_attribution_fraction >= 0.90
C0_false_positive_attribution_fraction >= 0.90
oracle_miss_attribution_fraction >= 0.90
```

and:

```text
at least one backfillable mode has coverage >= 0.003
```

### 可视化

```text
p1_C0_T2_C4_E2_venn.svg
p1_T2_removed_mode_sankey.svg
p1_E2_added_mode_sankey.svg
p1_oracle_miss_repairable_modes.svg
p1_feature_space_C0_T2_C4_E2.svg
```

---

## P2：C0 fine trim and minimal deletion

### 目标

验证 C0 broad seed 是否能通过更细粒度 trim 直接过 official gate。

### 必须记录

```text
trim_id
base_controller
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
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
removed_mode_distribution
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

C0 fine trim pass：

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
p2_C0_fine_trim_frontier.svg
p2_trim_coverage_loss_vs_bad_ucb.svg
p2_removed_mode_distribution.svg
p2_C0_trim_family_heatmap.svg
```

---

## P3：T2 coverage backfill

### 目标

从 T2 safe core 出发，最小回填 coverage deficit，同时不破坏 bad/null/confidence gate。

### 必须记录

```text
backfill_id
base_core = T2
candidate_pool
ranking_score
rows_added
coverage_before
coverage_after
precision_after
bad_event_after
null_rate_after
precision_lcb
bad_event_ucb
added_mode_distribution
added_family_distribution
accepted_strata_count
accepted_family_count
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

T2 backfill pass：

$$
Coverage_{\text{heldout}}\geq0.03,
$$

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
NullRate_{\text{heldout}}\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

Backfill efficiency diagnostic：

$$
RowsAdded \leq 0.25 \cdot RowsRemoved_{T2}.
$$

### 可视化

```text
p3_T2_backfill_frontier.svg
p3_backfill_rows_mode_sankey.svg
p3_coverage_deficit_fill_curve.svg
p3_T2_backfill_lcb_ucb.svg
```

---

## P4：C4/E2 filtered expansion

### 目标

把 E2 从 high-recall contaminated expansion 变成 usable expansion pool。

### 必须记录

```text
expansion_id
base_seed = C4
candidate_pool = E2
filter_features
thresholds
rows_added
coverage_after
precision_after
bad_event_after
null_rate_after
precision_lcb
bad_event_ucb
Jaccard
added_mode_distribution
accepted_strata_count
accepted_family_count
max_family_share
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Filtered expansion pass：

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

Diagnostic pass：

$$
Jaccard_{\text{legal,oracle}}\geq0.30
$$

and:

$$
BadEventRate\leq0.10
$$

with:

$$
Coverage\geq0.03.
$$

### 可视化

```text
p4_C4_filtered_expansion_frontier.svg
p4_E2_contamination_filter_curve.svg
p4_expansion_jaccard_vs_bad_event.svg
p4_C4_expansion_family_balance.svg
```

---

## P5：Constrained bridge fill-up

### 目标

组合 P2/P3/P4，构造 final bridge controller。

### 必须记录

```text
bridge_id
trim_id
backfill_id
expansion_id
fillup_strategy
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
oracle_gap_remaining
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
NullRate_{\text{heldout}}\leq0.15,
$$

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
p5_bridge_fillup_frontier.svg
p5_bridge_ladder_C0_T2_C4_E2.svg
p5_bridge_oracle_gap_closure.svg
p5_bridge_support_balance.svg
```

---

## P6：exact-reference deployable frontier v11

### 目标

在 CBD0 exact reference 上确认最终 bridge frontier 是否可部署。

### 必须记录

```text
controller_id
bridge_id
trim_id
backfill_id
expansion_id
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

If no controller passes, route must be one of:

```text
R10-C0FineTrimStillFails
R11-T2BackfillFails
R12-E2FilteredExpansionFails
R13-BridgeFillupStillFails
R14-ExactReferenceStillNotDeployableAfterLocalizedBridge
```

### 可视化

```text
p6_reference_frontier_v11_precision_coverage_bad_null.svg
p6_lcb_ucb_confidence_frontier.svg
p6_oracle_legal_gap_after_localized_bridge.svg
p6_deployable_region_ladder.svg
```

---

## P7：true-delta compute v12 parallel lane

### 目标

继续推进 system path。若 P6 exact-reference pass，P7 变为 primary blocker；若 P6 不过，P7 只作 parallel diagnostic。

### 必须记录

```text
custom_delta_id
layout
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
AUC_safe_good
AUC_safe_useful
AUC_bridge_accept
corr_safe_grounded
agreement_exact_accept
step_ratio_q90
memory_ratio
dominant_residual_subphase
bridge_score_component_time
trim_component_time
backfill_component_time
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
p7_true_delta_v12_cost_signal_pareto.svg
p7_compute_vs_decision_ladder.svg
p7_residual_subphase_after_localized_bridge.svg
```

---

## P8：system-legal exact-signal controller

### 目标

只有 P6 reference deployable pass 与 P7 compute pass 同时成立时，建立 official controller。

### 必须记录

```text
controller_id
custom_delta_id
bridge_id
trim_id
backfill_id
expansion_id
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
p8_system_controller_precision_coverage_bad_null.svg
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
bridge_id
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
bridge_id = best P9 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledBridgeController, ShuffledTrimScore,
           ShuffledBackfillScore, ShuffledExpansionScore, ShuffledSupportStat,
           ShuffledControlGain, ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledSignalChannel, FunctionalChannelShuffled, TailMaskShuffled,
           RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled,
           InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
bridge_id
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
ShuffledTrimScore = fail
ShuffledBackfillScore = fail
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
contract_audit_v9267.csv
p0_v9266_boundary_reproduction.csv
p1_C0_T2_C4_E2_autopsy.csv
p2_C0_fine_trim_minimal_deletion.csv
p3_T2_coverage_backfill.csv
p4_C4_E2_filtered_expansion.csv
p5_constrained_bridge_fillup.csv
p6_exact_reference_deployable_frontier_v11.csv
p7_true_delta_compute_v12_parallel_lane.csv
p8_system_legal_exact_signal_controller.csv
p9_leave_dataset_and_stratum_out.csv
p10_official_paired_replay.csv
p11_short_run_functional_validation.csv
C0_T2_C4_E2_membership_trace_v9267.csv
T2_removed_rows_trace_v9267.csv
E2_added_rows_trace_v9267.csv
C0_trim_finegrid_trace_v9267.csv
T2_backfill_trace_v9267.csv
C4_filtered_expansion_trace_v9267.csv
bridge_fillup_trace_v9267.csv
reference_frontier_v11_trace.csv
true_delta_compute_v12_trace.csv
system_controller_trace_v9267.csv
leaveout_trace_v9267.csv
paired_replay_branch_trace_v9267.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9266_boundary_unstable
F3_dataset_tuning_detected
F4_T2_removed_autopsy_incomplete
F5_E2_added_autopsy_incomplete
F6_C0_false_positive_unattributed
F7_C0_fine_trim_fail
F8_C0_trim_coverage_loss
F9_T2_backfill_fail
F10_T2_backfill_bad_event_fail
F11_T2_backfill_null_fail
F12_E2_filtered_expansion_fail
F13_E2_filtered_expansion_bad_event_fail
F14_bridge_fillup_tiny
F15_bridge_precision_lcb_fail
F16_bridge_bad_event_ucb_fail
F17_bridge_null_rate_fail
F18_exact_reference_still_not_deployable_after_localized_bridge
F19_true_delta_compute_still_expensive
F20_true_delta_signal_lost
F21_system_controller_precision_fail
F22_system_controller_coverage_fail
F23_system_controller_bad_event_fail
F24_system_controller_null_rate_fail
F25_leave_dataset_out_fail
F26_leave_stratum_out_fail
F27_paired_replay_control_equivalent
F28_shuffle_control_pass
F29_functional_lr_equivalent
F30_short_run_task_drop
F31_full_run_no_macro_hard_stratum_gain
F32_strong_baseline_explains_gain
F33_robustness_fail
F34_external_not_ready
F35_fake_or_proxy_violation
F36_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.66 boundary reproduced.

R2-LocalizedAutopsyPass:
  C0/T2/C4/E2 modes are attributed.

R3-C0FineTrimPass:
  C0 broad seed becomes deployable after finer minimal trim.

R4-T2BackfillPass:
  T2 safe core recovers coverage without breaking bad/null/confidence.

R5-C4FilteredExpansionPass:
  C4/E2 expansion recovers deployable coverage after filtering.

R6-BridgeFillupPass:
  constrained bridge fill-up passes heldout frontier.

R7-ExactReferenceDeployableFrontierPass:
  CBD0 exact reference + localized bridge frontier passes precision / coverage / bad-event / null gate.

R8-TrueDeltaComputeV12SystemPass:
  true branch-delta v12 passes system envelope while preserving signal.

R9-SystemLegalExactSignalControllerPass:
  system-legal true-delta controller passes heldout gate.

R10-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R11-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R12-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R13-C0FineTrimStillFails:
  broad C0 seed cannot be made safe without losing coverage.

R14-T2BackfillFails:
  T2 safety core cannot recover required coverage.

R15-C4ExpansionStillContaminated:
  C4/E2 expansion cannot be filtered into deployable rows.

R16-BridgeStillTiny:
  bridge remains coverage < 0.03.

R17-ReferenceFeasibleButComputeFail:
  decision geometry viable, true-delta compute still too expensive.

R18-ComputePassButReferenceFail:
  compute viable, accept-region geometry still fails.

R19-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R20-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R21-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R22-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9266_boundary_pass
dataset_tuning_detected
localized_autopsy_pass
T2_removed_attribution_fraction
E2_added_attribution_fraction
C0_false_positive_attribution_fraction
oracle_miss_attribution_fraction
best_C0_trim_id
C0_fine_trim_pass
C0_trim_precision
C0_trim_coverage
C0_trim_bad_event
C0_trim_null_rate
C0_trim_precision_lcb
C0_trim_bad_event_ucb
best_T2_backfill_id
T2_backfill_pass
T2_backfill_precision
T2_backfill_coverage
T2_backfill_bad_event
T2_backfill_null_rate
best_C4_expansion_id
C4_filtered_expansion_pass
C4_expansion_precision
C4_expansion_coverage
C4_expansion_bad_event
C4_expansion_null_rate
best_bridge_id
bridge_fillup_pass
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
oracle_gap_remaining
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
success_v9267_strict_purekan_functional
success_v9267_full_functional
success_v9267_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 C0/T2/C4/E2 autopsy
  P2 C0 fine trim
  P3 T2 coverage backfill
  P4 C4/E2 filtered expansion
  P5 constrained bridge fill-up
  P6 exact reference deployable frontier
  P7 true-delta compute v12

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
P2/P3/P4 can run in parallel.
P5 bridge can use best survivors from P2/P3/P4.
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
v9.2.66 boundary reproduced
C0/T2/C4/E2 autopsy completed
C0 fine trim measured
T2 backfill measured
C4/E2 filtered expansion measured
bridge fill-up measured
exact reference frontier v11 measured
true-delta compute v12 measured
no fake/proxy/offload/loss/teacher violation
```

## Decision geometry success

```text
Minimum diagnostic success
+
bridge fill-up pass
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
1. v9.2.66 boundary cannot be reproduced；
2. C0/T2/C4/E2 autopsy cannot attribute major modes；
3. C0 fine trim cannot fix bad-event without coverage loss；
4. T2 backfill cannot recover coverage；
5. T2 backfill raises null-rate above 0.15；
6. E2 filtered expansion remains contaminated；
7. bridge fill-up remains coverage < 0.03；
8. precision LCB remains below 0.75；
9. bad-event UCB remains above 0.05；
10. null rate remains above 0.15；
11. exact reference remains not deployable after localized bridge；
12. true branch-delta compute remains too expensive；
13. true branch-delta compute passes system but loses signal；
14. system-legal controller cannot meet precision / coverage / bad-event / null-rate；
15. leave-dataset-out fails；
16. leave-stratum-out fails；
17. paired replay remains control-equivalent；
18. shuffle controls pass, indicating overfit；
19. short-run task drops；
20. full run gives no macro / hard-stratum / geometry gain；
21. functional breaks system gate；
22. gains are explained by QuadraticFeatureMLP；
23. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：C0 fine trim passes

可以声明：

```text
C0 broad seed was viable; previous failure was coarse trim / confidence calibration.
```

然后进入 P6 exact reference and P7 true-delta compute。

## Case B：T2 backfill passes

可以声明：

```text
T2 safety core was viable; previous failure was over-trim coverage deficit.
```

然后进入 exact-reference frontier and system lane。

## Case C：C4 filtered expansion passes

可以声明：

```text
C4 seed can recover coverage through filtered legal-neighbor expansion.
```

但仍需 exact-reference frontier and LDO/LSO。

## Case D：bridge passes but compute fails

必须声明：

```text
decision geometry is viable, but true branch-delta system implementation remains blocker.
```

下一步继续 kernel / layout / fused bridge-score compute，不调 dataset。

## Case E：compute passes but bridge fails

必须声明：

```text
value is observable and cheap, but accept/abstain geometry is not deployable.
```

下一步继续修 C0/T2/C4/E2 localized decision geometry，不走 kernel-only。

## Case F：all localized repairs fail

必须声明：

```text
oracle frontier exists, but current legal features cannot bridge broad-borderline and precise-tiny frontiers.
```

下一步应重设 legal sufficient statistics / output-delta features，而不是继续 threshold tuning。

## Case G：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

---

# Part XII. 最终建议

v9.2.67 的一句话策略是：

$$
\boxed{
\text{不要再造全局 score；围绕 C0/T2/C4/E2 做局部化 bridge repair：少删、回填、过滤、约束填充。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
support 是否够；
joint score 是否有 AUC；
conditional null/bad 是否有信号；
true-delta exact signal 是否存在；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. T2 删除的 177 rows 里有多少是 over-trim safe-useful？
2. 能否只回填 coverage deficit 0.003876，而不破 bad/null/confidence？
3. C0 的 bad-event excess 是否可由更精细 local pocket trim 消除？
4. E2 的 2388 added rows 是否有可过滤的 safe-useful subset？
5. C0 trim、T2 backfill、C4 filtered expansion 能否组成 bridge？
6. exact-reference frontier 能否从 C0/T2/C4/E2 的分裂状态进入 deployable gate？
7. bridge exact-reference 过线后，true-delta compute 是否成为唯一 blocker？
8. system-legal controller 能否 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.67 的结果将给出清晰分叉：

```text
if C0 fine trim passes:
  promote C0-trim to exact-reference frontier.

if T2 backfill passes:
  use T2 as safety core and proceed.

if C4 filtered expansion passes:
  use C4/E2 as coverage recovery path.

if bridge passes but compute fails:
  kernelization is primary blocker.

if bridge fails:
  current legal features cannot recover oracle frontier; redesign sufficient statistics.

if reference and compute both pass:
  open system-legal controller, LDO/LSO, paired replay.
```
